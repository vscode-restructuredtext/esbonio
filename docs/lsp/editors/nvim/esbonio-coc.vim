"  --------------- First time setup ------------------
"  There are a few steps you need to perform when setting this up for the
"  first time.
"
"  1. Ensure you have vim-plug's `plug.vim` file installed in your autoload
"     directory. See https://github.com/junegunn/vim-plug#installation for
"     details.
"  2. Open a terminal in the directory containing this file and run the
"     following command to load this config isolated from your existing
"     configuration.
"
"        (n)vim -u esbonio-coc.vim
"  3. Install the coc.nvim plugin.
"
"     :PlugInstall
"
"  4. Install the coc-esbonio extension.
"
"     :CocInstall coc-esbonio
"
"  --------------- Subsequent use --------------------
"
"  1. Open a terminal in the directory containing this file and run the
"     following command to load it.
"
"     (n)vim -u esbonio-coc.vim

set expandtab
set tabstop=3
set softtabstop=3
set shiftwidth=3

call plug#begin('./plugins')

Plug 'neoclide/coc.nvim', {'branch': 'release'}

call plug#end()
