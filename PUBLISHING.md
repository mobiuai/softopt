# Publishing SoftOpt — step by step (all via terminal)

You're right, all of this can be done from the terminal. Two separate
things: (1) GitHub (the source code, README people read on the repo
page), (2) PyPI (the installable package `pip install softopt` pulls).
They don't depend on each other technically, but do (1) first since (2)'s
`pyproject.toml` links to it.

## 1. GitHub

```bash
cd softopt_package
git init
git add .
git commit -m "Initial commit: SoftOpt v0.1.0"
```

Create the empty repo on GitHub itself (there's no pure-terminal way to
create the repo on GitHub's servers without the `gh` CLI tool — install
it once, or create the empty repo by hand on github.com and skip to the
`git remote add` line):

```bash
brew install gh          # macOS, one-time
gh auth login             # one-time, follow the prompts
gh repo create softopt --public --source=. --remote=origin
git push -u origin main
```

If you created the repo by hand on github.com instead:
```bash
git remote add origin https://github.com/mobiuai/softopt.git
git branch -M main
git push -u origin main
```

**Before pushing**, update two placeholder URLs to your real username:
- `pyproject.toml` — the three `[project.urls]` lines
- `README.md` and `website/index.html` — the GitHub link in the footer

## 2. PyPI

Two accounts needed, both free: one on pypi.org itself, and (recommended)
an API token instead of your password.

```bash
pip install twine --upgrade
```

1. Create an account at https://pypi.org/account/register/
2. Go to https://pypi.org/manage/account/token/ and create an API token
   scoped to your whole account (you can narrow it to just this project
   after the first upload). Copy it — it starts with `pypi-`.
3. From `softopt_package/`, with `dist/` already built (it's in the
   package you have — `softopt-0.1.0.tar.gz` and the `.whl`):

```bash
twine upload dist/*
```

It'll ask for a username and password:
- username: `__token__` (literally that string)
- password: the `pypi-...` token you copied

That's it — `pip install softopt` works for everyone within a minute or
two. If you change the code later and want to publish an update, bump
the version number in `pyproject.toml` (e.g. `0.1.0` → `0.1.1`), rebuild,
and upload again:

```bash
pip install build --upgrade
rm -rf dist build
python3 -m build
twine upload dist/*
```

## 3. (Optional) test on a throwaway index first

If you want a dry run before the real upload, PyPI has a separate test
server:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ softopt
```

You'll need a *separate* free account at https://test.pypi.org (it does
not share accounts with the real pypi.org).

## 4. The website — softopt.mobiu.ai

`website/index.html` is a single, self-contained file — no build step.
Using GitHub Pages with your custom subdomain:

```bash
mkdir -p docs
cp website/index.html docs/
echo "softopt.mobiu.ai" > docs/CNAME
git add docs && git commit -m "Add landing page" && git push
```

Then two places to configure:

1. **On github.com**: repo → Settings → Pages → Source → "Deploy from a
   branch" → branch `main`, folder `/docs` → Save. GitHub will show you
   its own default address (something like `mobiuai.github.io/softopt`)
   — ignore that, the CNAME file is what makes your custom domain work
   instead.
2. **Wherever mobiu.ai's DNS is managed** (not GitHub — your domain
   registrar or DNS provider for mobiu.ai): add a CNAME record:
   - Host/name: `softopt`
   - Value/target: `mobiuai.github.io`

DNS propagation can take anywhere from a few minutes to a few hours.
Once it resolves, `https://softopt.mobiu.ai` serves `docs/index.html`.
If you ever restructure the repo (e.g. the site ends up in a different
folder), the only two things that matter are: the Pages source folder
in step 1 matches where `index.html` actually lives, and the `CNAME`
file sits in that same folder.
