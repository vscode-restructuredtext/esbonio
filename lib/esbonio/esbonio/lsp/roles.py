"""Role support."""
import json
import re
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pkg_resources
from pygls.lsp.types import CompletionItem
from pygls.lsp.types import CompletionItemKind
from pygls.lsp.types import Location
from pygls.lsp.types import MarkupContent
from pygls.lsp.types import MarkupKind
from pygls.lsp.types import Position
from pygls.lsp.types import Range
from pygls.lsp.types import TextEdit
from pygls.workspace import Document

from esbonio.lsp.directives import DIRECTIVE
from esbonio.lsp.rst import CompletionContext
from esbonio.lsp.rst import LanguageFeature
from esbonio.lsp.rst import RstLanguageServer
from esbonio.lsp.sphinx import SphinxLanguageServer

try:
    from typing import Protocol
except ImportError:
    # Protocol is only available in Python 3.8+
    class Protocol:  # type: ignore
        ...


ROLE = re.compile(
    r"""
    ([^\w:]|^\s*)                     # roles cannot be preceeded by letter chars
    (?P<role>
      :                               # roles begin with a ':' character
      (?!:)                           # the next character cannot be a ':'
      ((?P<domain>[\w]+):(?=\w))?     # roles may include a domain (that must be followed by a word character)
      ((?P<name>[\w-]+):?)?           # roles have a name
    )
    (?P<target>
      `                               # targets begin with a '`' character
      ((?P<alias>[^<`>]*?)<)?         # targets may specify an alias
      (?P<modifier>[!~])?             # targets may have a modifier
      (?P<label>[^<`>]*)?             # targets contain a label
      >?                              # labels end with a '>' when there's an alias
      `?                              # targets end with a '`' character
    )?
    """,
    re.VERBOSE,
)
"""A regular expression to detect and parse parial and complete roles.

I'm not sure if there are offical names for the components of a role, but the
language server breaks a role down into a number of parts::

                 vvvvvv label
                v modifier(optional)
               vvvvvvvv target
   :c:function:`!malloc`
   ^^^^^^^^^^^^ role
      ^^^^^^^^ name
    ^ domain (optional)

The language server sometimes refers to the above as a "plain" role, in that the
role's target contains just the label of the object it is linking to. However it's
also possible to define "aliased" roles, where the link text in the final document
is overriden, for example::

                vvvvvvvvvvvvvvvvvvvvvvvv alias
                                          vvvvvv label
                                         v modifier (optional)
               vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv target
   :c:function:`used to allocate memory <~malloc>`
   ^^^^^^^^^^^^ role
      ^^^^^^^^ name
    ^ domain (optional)

See :func:`tests.test_roles.test_role_regex` for a list of example strings this pattern
is expected to match.
"""


DEFAULT_ROLE = re.compile(
    r"""
    (?<![:`])
    (?P<target>
      `                               # targets begin with a '`' character
      ((?P<alias>[^<`>]*?)<)?         # targets may specify an alias
      (?P<modifier>[!~])?             # targets may have a modifier
      (?P<label>[^<`>]*)?             # targets contain a label
      >?                              # labels end with a '>' when there's an alias
      `?                              # targets end with a '`' character
    )
    """,
    re.VERBOSE,
)
"""A regular expression to detect and parse parial and complete "default" roles.

A "default" role is the target part of a normal role - but without the ``:name:`` part.
"""


class TargetDefinition(Protocol):
    """A definition provider for role targets"""

    def find_definitions(
        self, doc: Document, match: "re.Match", name: str, domain: Optional[str] = None
    ) -> List[Location]:
        """Return a list of locations representing the definition of the given role
        target.

        Parameters
        ----------
        doc:
           The document containing the match
        match:
           The match object that triggered the definition request
        name:
           The name of the role
        domain:
           The domain the role is part of, if applicable.
        """


class TargetCompletion(Protocol):
    """A completion provider for role targets"""

    def complete_targets(
        self, context: CompletionContext, domain: str, name: str
    ) -> List[CompletionItem]:
        """Return a list of completion items representing valid targets for the given
        role.

        Parameters
        ----------
        context:
           The completion context
        domain:
           The name of the domain the role is a member of
        name:
           The name of the role to generate completion suggestions for.
        """


class Roles(LanguageFeature):
    """Role support for the language server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._documentation: Dict[str, Dict[str, str]] = {}
        """Cache for documentation."""

        self._target_definition_providers: List[TargetDefinition] = []
        """A list of providers that locate the definition for the given role target."""

        self._target_completion_providers: List[TargetCompletion] = []
        """A list of providers that give completion suggestions for role target
        objects."""

    def add_target_definition_provider(self, provider: TargetDefinition) -> None:
        self._target_definition_providers.append(provider)

    def add_target_completion_provider(self, provider: TargetCompletion) -> None:
        self._target_completion_providers.append(provider)

    def add_documentation(self, documentation: Dict[str, Dict[str, Any]]) -> None:
        """Register role documentation.

        ``documentation`` should be a dictionary of the form ::

           documentation = {
               "raw(docutils.parsers.rst.roles.raw_role)": {
                   "is_markdown": true,
                   "license": "https://...",
                   "source": "https://...",
                   "description": [
                       "# :raw:",
                       "The raw role is used for...",
                       ...
                   ]
               }
           }

        where the key is of the form `name(dotted_name)`. There are cases where a role's
        implementation is not sufficient to uniquely identify it as multiple roles can
        be provided by a single class.

        This means the key has to be a combination of the ``name`` the user writes in
        an reStructuredText document and ``dotted_name`` is the fully qualified name of
        the role's implementation.

        .. note::

           If there is a clash with an existing key, the existing value will be
           overwritten with the new value.

        The values in this dictionary are themselves dictionaries with the following
        fields.

        ``description``
           A list of strings for the role's usage.

        ``is_markdown``
           A boolean flag used to indicate whether the ``description`` is written in
           plain text or markdown.

        ``source``
           The url to the documentation's source.

        ``license``
           The url to the documentation's license.

        Parameters
        ----------
        documentation:
           The documentation to register.
        """

        for key, doc in documentation.items():
            description = doc.get("description", [])
            if not description:
                continue

            source = doc.get("source", "")
            if source:
                description.append(f"\n[Source]({source})")

            license = doc.get("license", "")
            if license:
                description.append(f"\n[License]({license})")

            doc["description"] = "\n".join(description)
            self._documentation[key] = doc

    completion_triggers = [ROLE, DEFAULT_ROLE]
    definition_triggers = [ROLE]

    def definition(
        self, match: "re.Match", doc: Document, pos: Position
    ) -> List[Location]:

        groups = match.groupdict()
        domain = groups["domain"] or None
        name = groups["name"]

        definitions = []
        self.logger.debug(
            "Suggesting definitions for %s%s: %s",
            domain or ":",
            name,
            match.groupdict(),
        )

        for provide in self._target_definition_providers:
            definitions += provide.find_definitions(doc, match, name, domain) or []

        return definitions

    def complete(self, context: CompletionContext) -> List[CompletionItem]:
        """Generate completion suggestions relevant to the current context.

        This function is a little intense, but its sole purpose is to determine the
        context in which the completion request is being made and either return
        nothing, or the results of :meth:`~esbonio.lsp.roles.Roles.complete_roles` or
        :meth:`esbonio.lsp.roles.Roles.complete_targets` whichever is appropriate.

        Parameters
        ----------
        context:
           The context of the completion request.
        """

        # Do not suggest completions within the middle of Python code.
        if context.location == "py":
            return []

        groups = context.match.groupdict()
        target = groups["target"]

        # All text matched by the regex
        text = context.match.group(0)
        start, end = context.match.span()

        if target:
            target_index = start + text.find(target)

            # Only trigger target completions if the request was made from within
            # the target part of the role.
            if target_index <= context.position.character <= end:
                return self.complete_targets(context)

        # If there's no indent, then this can only be a role definition
        indent = context.match.group(1)
        if indent == "":
            return self.complete_roles(context)

        # Otherwise, search backwards until we find a blank line or an unindent
        # so that we can determine the appropriate context.
        linum = context.position.line - 1

        try:
            line = context.doc.lines[linum]
        except IndexError:
            return self.complete_roles(context)

        while linum >= 0 and line.startswith(indent):
            linum -= 1
            line = context.doc.lines[linum]

        # Unless we are within a directive's options block, we should offer role
        # suggestions
        if DIRECTIVE.match(line):
            return []

        return self.complete_roles(context)

    def completion_resolve(self, item: CompletionItem) -> CompletionItem:

        # We need extra info to know who to call
        if not item.data:
            return item

        data = typing.cast(Dict, item.data)
        ctype = data.get("completion_type", "")

        if ctype == "role":
            return self.completion_resolve_role(item)

        return item

    def complete_roles(self, context: CompletionContext) -> List[CompletionItem]:

        match = context.match
        groups = match.groupdict()
        domain = groups["domain"] or ""
        items = []

        # Insert text starting from the starting ':' character of the role.
        start = match.span()[0] + match.group(0).find(":")
        end = start + len(groups["role"])

        range_ = Range(
            start=Position(line=context.position.line, character=start),
            end=Position(line=context.position.line, character=end),
        )

        for name, role in self.rst.get_roles().items():

            if not name.startswith(domain):
                continue

            try:
                dotted_name = f"{role.__module__}.{role.__name__}"
            except AttributeError:
                dotted_name = f"{role.__module__}.{role.__class__.__name__}"

            insert_text = f":{name}:"
            item = CompletionItem(
                label=name,
                kind=CompletionItemKind.Function,
                detail=f"{dotted_name}",
                filter_text=insert_text,
                text_edit=TextEdit(range=range_, new_text=insert_text),
                data={"completion_type": "role"},
            )

            items.append(item)

        return items

    def completion_resolve_role(self, item: CompletionItem) -> CompletionItem:

        # We need the detail field set to the role implementation's fully qualified name
        if not item.detail:
            return item

        documentation = self.get_documentation(item.label, item.detail)
        if not documentation:
            return item

        description = documentation.get("description", "")
        is_markdown = documentation.get("is_markdown", False)
        kind = MarkupKind.Markdown if is_markdown else MarkupKind.PlainText

        item.documentation = MarkupContent(kind=kind, value=description)
        return item

    def complete_targets(self, context: CompletionContext) -> List[CompletionItem]:
        """Generate the list of role target completion suggestions."""

        groups = context.match.groupdict()

        # Handle the default role case.
        if "role" not in groups:
            domain, name = self.rst.get_default_role()
            if not name:
                return []
        else:
            name = groups["name"]
            domain = groups["domain"]

        domain = domain or ""
        name = name or ""

        # Only generate suggestions for "aliased" targets if the request comes from
        # within the <> chars.
        if groups["alias"]:
            text = context.match.group(0)
            start = context.match.span()[0] + text.find(groups["alias"])
            end = start + len(groups["alias"])

            if start <= context.position.character <= end:
                return []

        targets = []

        startchar = "<" if "<" in groups["target"] else "`"
        endchars = ">`" if "<" in groups["target"] else "`"

        start, end = context.match.span()
        start += context.match.group(0).index(startchar) + 1
        range_ = Range(
            start=Position(line=context.position.line, character=start),
            end=Position(line=context.position.line, character=end),
        )
        prefix = context.match.group(0)[start:]
        modifier = groups["modifier"] or ""

        for provide in self._target_completion_providers:
            candidates = provide.complete_targets(context, domain, name) or []

            for candidate in candidates:

                # Don't interfere with items that already carry a `text_edit`, allowing
                # some providers (like filepaths) to do something special.
                if not candidate.text_edit:
                    new_text = candidate.insert_text or candidate.label

                    # This is rather annoying, but `filter_text` needs to start with
                    # the text we are going to replace, otherwise VSCode won't show our
                    # suggestions!
                    candidate.filter_text = f"{prefix}{new_text}"

                    candidate.text_edit = TextEdit(
                        range=range_, new_text=f"{modifier}{new_text}"
                    )
                    candidate.insert_text = None

                if not candidate.text_edit.new_text.endswith(endchars):
                    candidate.text_edit.new_text += endchars

                targets.append(candidate)

        return targets

    def get_documentation(
        self, label: str, implementation: str
    ) -> Optional[Dict[str, Any]]:
        """Return the documentation for the given role, if available.

        If documentation for the given ``label`` cannot be found, this function will also
        look for the label under the project's :confval:`sphinx:primary_domain` followed
        by the ``std`` domain.

        Parameters
        ----------
        label:
           The name of the role, as the user would type in an reStructuredText file.
        implementation:
           The full dotted name of the role's implementation.
        """

        key = f"{label}({implementation})"
        documentation = self._documentation.get(key, None)
        if documentation:
            return documentation

        if not isinstance(self.rst, SphinxLanguageServer) or not self.rst.app:
            return None

        # Nothing found, try the primary domain
        domain = self.rst.app.config.primary_domain
        key = f"{domain}:{label}({implementation})"

        documentation = self._documentation.get(key, None)
        if documentation:
            return documentation

        # Still nothing, try the standard domain
        key = f"std:{label}({implementation})"

        documentation = self._documentation.get(key, None)
        if documentation:
            return documentation

        return None


def esbonio_setup(rst: RstLanguageServer):
    roles = Roles(rst)
    rst.add_feature(roles)

    docutils_docs = pkg_resources.resource_string("esbonio.lsp.rst", "roles.json")
    roles.add_documentation(json.loads(docutils_docs.decode("utf8")))

    if isinstance(rst, SphinxLanguageServer):
        sphinx_docs = pkg_resources.resource_string("esbonio.lsp.sphinx", "roles.json")
        roles.add_documentation(json.loads(sphinx_docs.decode("utf8")))
