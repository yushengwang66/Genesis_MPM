# Genesis_MPM

This repository is prepared for working with the official Genesis simulation platform for MPM-related development.

The official upstream project is:

- https://github.com/Genesis-Embodied-AI/genesis-world

In this repository, the official Genesis source is intended to be tracked as a Git submodule at `Genesis_MPM`.

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/yushengwang66/Genesis_MPM.git
cd Genesis_MPM
```

## If you already cloned without submodules

```bash
git submodule update --init --recursive
```

## Update upstream Genesis later

```bash
cd Genesis_MPM
git fetch origin
# choose the upstream branch or commit you want
cd ..
git add Genesis_MPM
git commit -m "Update Genesis_MPM upstream pointer"
git push
```
