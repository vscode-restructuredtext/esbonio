import enum
import importlib
import json
import logging
import os
from pathlib import Path, PurePath, PurePosixPath
import re
import textwrap
import traceback
from typing import Iterable, List, Tuple
from typing import Type

from pygls.lsp.methods import CODE_ACTION
from pygls.lsp.methods import COMPLETION
from pygls.lsp.methods import COMPLETION_ITEM_RESOLVE
from pygls.lsp.methods import DEFINITION
from pygls.lsp.methods import DOCUMENT_LINK_RESOLVE
from pygls.lsp.methods import DOCUMENT_SYMBOL
from pygls.lsp.methods import INITIALIZE
from pygls.lsp.methods import INITIALIZED
from pygls.lsp.methods import TEXT_DOCUMENT_DID_CHANGE
from pygls.lsp.methods import TEXT_DOCUMENT_DID_OPEN
from pygls.lsp.methods import TEXT_DOCUMENT_DID_SAVE
from pygls.lsp.methods import WORKSPACE_DID_DELETE_FILES
from pygls.lsp.types import CodeActionParams
from pygls.lsp.types import CompletionItem
from pygls.lsp.types import CompletionList
from pygls.lsp.types import CompletionOptions
from pygls.lsp.types import CompletionParams
from pygls.lsp.types import DefinitionParams
from pygls.lsp.types import DeleteFilesParams
from pygls.lsp.types import DidChangeTextDocumentParams
from pygls.lsp.types import DidOpenTextDocumentParams
from pygls.lsp.types import DidSaveTextDocumentParams
from pygls.lsp.types import DocumentLink
from pygls.lsp.types import DocumentSymbolParams
from pygls.lsp.types import FileOperationFilter
from pygls.lsp.types import FileOperationPattern
from pygls.lsp.types import FileOperationRegistrationOptions
from pygls.lsp.types import InitializedParams
from pygls.lsp.types import InitializeParams
from pygls.lsp.types import ServerCapabilities
from pygls.protocol import LanguageServerProtocol

from .rst import CompletionContext, DocumentLinkContext
from .rst import DefinitionContext
from .rst import LanguageFeature
from .rst import RstLanguageServer
from .rst import SymbolVisitor

__version__ = "0.9.0"

__all__ = [
    "LanguageFeature",
    "RstLanguageServer",
    "create_language_server",
]

logger = logging.getLogger(__name__)


class Patched(LanguageServerProtocol):
    """Tweaked version of the protocol allowing us to tweak how the `ServerCapabilities`
    are constructed."""

    def __init__(self, *args, **kwargs):
        self._server_capabilities = ServerCapabilities()
        super().__init__(*args, **kwargs)

    @property
    def server_capabilities(self):
        return self._server_capabilities

    @server_capabilities.setter
    def server_capabilities(self, value: ServerCapabilities):

        if WORKSPACE_DID_DELETE_FILES in self.fm.features:
            opts = self.fm.feature_options.get(WORKSPACE_DID_DELETE_FILES, None)
            if opts:
                value.workspace.file_operations.did_delete = opts  # type: ignore

        self._server_capabilities = value


def create_language_server(
    server_cls: Type[RstLanguageServer], modules: Iterable[str], *args, **kwargs
) -> RstLanguageServer:
    """Create a new language server instance.

    Parameters
    ----------
    server_cls:
       The class definition to create the server from.
    modules:
       The list of modules that should be loaded.
    args, kwargs:
       Any additional arguments that should be passed to the language server's
       constructor.
    """

    if "logger" not in kwargs:
        kwargs["logger"] = logger

    server = server_cls(*args, **kwargs, protocol_cls=Patched)

    for module in modules:
        _load_module(server, module)

    return _configure_lsp_methods(server)


def _configure_lsp_methods(server: RstLanguageServer) -> RstLanguageServer:
    @server.feature(INITIALIZE)
    def on_initialize(ls: RstLanguageServer, params: InitializeParams):
        ls.initialize(params)

        for feature in ls._features.values():
            feature.initialize(params)

    @server.feature(INITIALIZED)
    def on_initialized(ls: RstLanguageServer, params: InitializedParams):
        ls.initialized(params)

        for feature in ls._features.values():
            feature.initialized(params)

    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    def on_open(ls: RstLanguageServer, params: DidOpenTextDocumentParams):
        pass

    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    def on_change(ls: RstLanguageServer, params: DidChangeTextDocumentParams):
        pass

    @server.feature(TEXT_DOCUMENT_DID_SAVE)
    def on_save(ls: RstLanguageServer, params: DidSaveTextDocumentParams):
        ls.save(params)

        for feature in ls._features.values():
            feature.save(params)

    @server.feature(
        WORKSPACE_DID_DELETE_FILES,
        FileOperationRegistrationOptions(
            filters=[
                FileOperationFilter(
                    pattern=FileOperationPattern(glob="**/*.rst"),
                )
            ]
        ),
    )
    def on_delete_files(ls: RstLanguageServer, params: DeleteFilesParams):
        ls.delete_files(params)

        for feature in ls._features.values():
            feature.delete_files(params)

    @server.feature(CODE_ACTION)
    def on_code_action(ls: RstLanguageServer, params: CodeActionParams):
        actions = []

        for feature in ls._features.values():
            actions += feature.code_action(params)

        return actions

    # <engine-example>
    @server.feature(
        COMPLETION,
        CompletionOptions(
            trigger_characters=[".", ":", "`", "<", "/"], resolve_provider=True
        ),
    )
    def on_completion(ls: RstLanguageServer, params: CompletionParams):
        uri = params.text_document.uri
        pos = params.position

        doc = ls.workspace.get_document(uri)
        line = ls.line_at_position(doc, pos)
        location = ls.get_location_type(doc, pos)

        items = []

        for name, feature in ls._features.items():
            for pattern in feature.completion_triggers:
                for match in pattern.finditer(line):
                    if not match:
                        continue

                    # Only trigger completions if the position of the request is within
                    # the match.
                    start, stop = match.span()
                    if start <= pos.character <= stop:
                        context = CompletionContext(
                            doc=doc, location=location, match=match, position=pos
                        )
                        ls.logger.debug("Completion context: %s", context)

                        for item in feature.complete(context):
                            item.data = {"source_feature": name, **(item.data or {})}
                            items.append(item)

        return CompletionList(is_incomplete=False, items=items)

    # </engine-example>

    @server.feature(COMPLETION_ITEM_RESOLVE)
    def on_completion_resolve(
        ls: RstLanguageServer, item: CompletionItem
    ) -> CompletionItem:
        source = (item.data or {}).get("source_feature", "")
        feature = ls.get_feature(source)

        if not feature:
            ls.logger.error(
                "Unable to resolve completion item, unknown source: '%s'", source
            )
            return item

        return feature.completion_resolve(item)

    @server.feature(DEFINITION)
    def on_definition(ls: RstLanguageServer, params: DefinitionParams):
        uri = params.text_document.uri
        pos = params.position

        doc = ls.workspace.get_document(uri)
        line = ls.line_at_position(doc, pos)
        location = ls.get_location_type(doc, pos)

        definitions = []

        for feature in ls._features.values():
            for pattern in feature.definition_triggers:
                for match in pattern.finditer(line):
                    if not match:
                        continue

                    start, stop = match.span()
                    if start <= pos.character and pos.character <= stop:
                        context = DefinitionContext(
                            doc=doc, location=location, match=match, position=pos
                        )
                        definitions += feature.definition(context)

        return definitions

    @server.feature(DOCUMENT_LINK_RESOLVE)
    def on_document_link_resolve(ls: RstLanguageServer, params: DocumentLink):
        fileName = params.target
        data = DocumentLinkContext(params.data)
        docPath = data.fileName
        resolveType = data.type
        if resolveType == "doc":
            resolved_target_path = add_doc_target_ext(
                fileName, PurePath(docPath), Path(ls.workspace.root_path)
            )
            if os.path.exists(resolved_target_path):
                ls.logger.error("found doc")
                return str(resolved_target_path)
            ls.logger.error("resolved path does not exist: '%s'", resolved_target_path)
            return None
        elif resolveType == "directive":
            resolved_target_path = add_directive_target(
                fileName, PurePath(docPath), Path(ls.workspace.root_path)
            )
            if os.path.exists(resolved_target_path):
                ls.logger.error("found directive")
                return str(resolved_target_path)
            ls.logger.error("resolved path does not exist: '%s'", resolved_target_path)
            return None
        else:
            ls.logger.error("resolveType is not supported: '%s'", resolveType)
            return None

    @server.feature(DOCUMENT_SYMBOL)
    def on_document_symbol(ls: RstLanguageServer, params: DocumentSymbolParams):

        doctree = ls.get_doctree(uri=params.text_document.uri)
        if doctree is None:
            return []

        visitor = SymbolVisitor(ls, doctree)
        doctree.walkabout(visitor)

        return visitor.symbols

    @server.command("esbonio.server.configuration")
    def get_configuration(ls: RstLanguageServer, *args):
        """Get the server's configuration.

        Not to be confused with the ``workspace/configuration`` request where the server
        can request the client's configuration. This is so client's can ask for sphinx's
        output path for example.

        As far as I know, there isn't anything built into the spec to cater for this?
        """
        config = ls.configuration
        ls.logger.debug("%s: %s", "esbonio.server.configuration", config)

        return config

    return server


def _load_module(server: RstLanguageServer, module: str):

    try:
        mod = importlib.import_module(module)
    except ImportError:
        logger.error("Unable to import module '%s'\n%s", module, traceback.format_exc())
        return

    if not hasattr(mod, "esbonio_setup"):
        logger.error(
            "Unable to load module '%s', missing 'esbonio_setup' function", module
        )
        return

    try:
        mod.esbonio_setup(server)
    except Exception:
        logger.error(
            "Error while setting up module '%s'\n%s", module, traceback.format_exc()
        )


def dump(obj) -> str:
    """Debug helper function that converts an object to JSON."""

    def default(o):
        if isinstance(o, enum.Enum):
            return o.value

        fields = {}
        for k, v in o.__dict__.items():

            if v is None:
                continue

            # Truncate long strings - but not uris!
            if isinstance(v, str) and not k.lower().endswith("uri"):
                v = textwrap.shorten(v, width=25)

            fields[k] = v

        return fields

    return json.dumps(obj, default=default)

class FileId(PurePosixPath):
    """An unambiguous file path relative to the local project's root."""

    PAT_FILE_EXTENSIONS = re.compile(r"\.((txt)|(rst)|(yaml))$")

    def collapse_dots(self) -> "FileId":
        result: List[str] = []
        for part in self.parts:
            if part == "..":
                result.pop()
            elif part == ".":
                continue
            else:
                result.append(part)
        return FileId(*result)

    @property
    def without_known_suffix(self) -> str:
        """Returns the fileid without any of its known file extensions (txt, rst, yaml)"""
        fileid = self.with_name(self.PAT_FILE_EXTENSIONS.sub("", self.name))
        return fileid.as_posix()

    def as_dirhtml(self) -> str:
        """Return a path string usable for referring to this page under the dirhtml static
        site convention."""

        # The project root is special
        if self == FileId("index.txt"):
            return ""

        return self.without_known_suffix + "/"

def reroot_path(
    filename: PurePosixPath, docpath: PurePath, project_root: Path
) -> Tuple[FileId, Path]:
    """Files within a project may refer to other files. Return a canonical path
    relative to the project root."""
    if filename.is_absolute():
        rel_fn = FileId(*filename.parts[1:])
    else:
        rel_fn = FileId(*docpath.parent.joinpath(filename).parts).collapse_dots()
    try:
        return rel_fn, project_root.joinpath(rel_fn).resolve()
    except ValueError:
        return rel_fn, Path(filename)

RST_EXTENSIONS = {".rst"}

def add_doc_target_ext(target: str, docpath: PurePath, project_root: Path) -> Path:
    """Given the target file of a doc role, add the appropriate extension and return full file path"""
    # Add .txt or .rst to end of doc role target path
    target_path = PurePosixPath(target)
    if target.endswith("/"):
        # return directly if target is a folder.
        fileid, resolved_target_path = reroot_path(target_path, docpath, project_root)
        return resolved_target_path
    # File already exists, like images
    fileid, resolved_target_path = reroot_path(target_path, docpath, project_root)
    if os.path.exists(resolved_target_path):
        return resolved_target_path
    # Adding the current suffix first takes into account dotted targets
    for ext in RST_EXTENSIONS:
        new_suffix = target_path.suffix + ext
        temp_path = target_path.with_suffix(new_suffix)

        fileid, resolved_target_path_suffix = reroot_path(
            temp_path, docpath, project_root
        )
        if os.path.exists(resolved_target_path_suffix):
            return resolved_target_path_suffix
    # If none of the files exists, return the original file path to trigger errors.
    return resolved_target_path

def add_directive_target(target: str, docpath: PurePath, project_root: Path) -> Path:
    """Given the target file of a directive and return full file path"""

    target_path = PurePosixPath(target)
    fileid, resolved_target_path = reroot_path(target_path, docpath, project_root)
    return resolved_target_path
