# OpenSCAD: local Linux GUI and remote command line

Run the installation steps below in a terminal on your **local Linux desktop**,
not an SSH session or a VS Code Remote terminal. The remote server's `openscad`
shortcut does not install anything on your local computer.

## Install locally

These commands reproduce the official **2026.11.01 development snapshot** used
on the server. This is a pinned development build, not a stable release or a
promise that it will remain the newest version.

Check your local architecture:

```bash
uname -m
```

Continue if it prints `x86_64` (Intel/AMD 64-bit). For `aarch64`/ARM, choose a
compatible build from the [official downloads page](https://openscad.org/downloads.html#snapshots)
instead; the file below is x86-64 only. You need `curl` and `sha256sum` installed.

Paste this block into your local terminal. It downloads the AppImage and its
published checksum, verifies the download, and makes it executable. No sudo is
needed, and the application stays outside any project checkout.

```bash
(
  set -eu
  mkdir -p "$HOME/Applications/OpenSCAD"
  cd "$HOME/Applications/OpenSCAD"
  file=OpenSCAD-2026.11.01-x86_64.AppImage
  curl -fL --retry 3 -O "https://files.openscad.org/snapshots/$file"
  curl -fL --retry 3 -O "https://files.openscad.org/snapshots/$file.sha256"
  sha256sum -c "$file.sha256"
  chmod u+x "$file"
  "./$file" --appimage-extract-and-run --version
)
```

Expect a checksum result of `OK` and `OpenSCAD version 2026.11.01`. If either
fails, stop and inspect the error. The extraction flag avoids needing FUSE;
it worked on the remote machine, where normal AppImage mounting failed.

## Add the short command

This creates `~/.local/bin/openscad`. If you already have a custom file at that
path, preserve it and use a different shortcut name instead.

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/openscad" <<'EOF'
#!/bin/sh
exec "$HOME/Applications/OpenSCAD/OpenSCAD-2026.11.01-x86_64.AppImage" --appimage-extract-and-run "$@"
EOF
chmod u+x "$HOME/.local/bin/openscad"
export PATH="$HOME/.local/bin:$PATH"
openscad --version
```

Then open the GUI with:

```bash
openscad
```

Or open a local model directly with `openscad /path/to/assembly.scad`.
If a new terminal reports `openscad: command not found`, add
`export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` for Bash or `~/.zshrc`
for Zsh, then open another terminal.

For an application-menu shortcut, run this locally after creating the wrapper:

```bash
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/openscad-nightly.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=OpenSCAD Nightly
Exec="$HOME/.local/bin/openscad" %f
Icon=applications-graphics
Terminal=false
Categories=Graphics;3DGraphics;Engineering;
EOF
```

Search your desktop's application menu for **OpenSCAD Nightly**.
The official downloads page recommends the Manifold geometry engine for faster
rendering; select it in Preferences if it is not already active.

## Copy a model from the server

The local application needs local copies of the model and its includes. For an
instrument, download the complete `openscad/` folder, including `assembly.scad`,
`parameters_native.scad`, and `common.scad`; open **assembly.scad**.

For example, run this on your local computer, replacing `YOUR_SSH_HOST` with the
SSH hostname/alias you normally use and `/path/to/repository` with its checkout path.
Choose a new destination for the first copy.

```bash
mkdir -p "$HOME/Downloads"
scp -r YOUR_SSH_HOST:/path/to/repository/output/objects/hkqai-instruments/cary60/codex/products/openscad \
  "$HOME/Downloads/cary60-openscad"
openscad "$HOME/Downloads/cary60-openscad/assembly.scad"
```

Alternatively, download and extract `output/packages/instruments_share.zip` if
available. See the [instrument guide](../instruments/README.md) for creating it.
Local edits do not automatically sync back to the server or regenerate GLB/URDF
files; use that guide's parameter import/rebuild workflow when needed.

## Headless use

Native CAD builds use `openscad` on PATH or the `OPENSCAD` executable override.
No GUI is required for geometry checks and STL exports:

```bash
openscad --version
.venv/bin/python -m src.instruments stl --model cary60 --part socket_left
```

Keep the selected executable installed while builds depend on it. Local server
installation details belong in the optional, Git-ignored `LOCAL.md`.
