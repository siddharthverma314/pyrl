{
  inputs = {
    cpprb = { url = github:ymd-h/cpprb; flake = false; };
    flatten-dict = { url = github:ianlini/flatten-dict; flake = false; };
    gitignore = { url = github:hercules-ci/gitignore; flake = false; };
    mujoco = { url = https://www.roboti.us/download/mjpro150_linux.zip; flake = false; };
    mujoco-py = { url = github:siddharthverma314/mujoco-py; flake = false; };
    pyGLFW = { url = github:FlorianRhiem/pyGLFW; flake = false; };
    scikit-video = { url = github:scikit-video/scikit-video; flake = false; };
    nixpkgs.url = github:NixOS/nixpkgs/nixpkgs-unstable;
  };
  outputs = inputs: let
    mkPkg = cudaSupport: python: rec {
      overlay = import ./nix/overlay.nix { inherit cudaSupport python inputs; };
      pkgs = import inputs.nixpkgs {
        overlays = [ overlay ];
        system = "x86_64-linux";
        config = { inherit cudaSupport; allowUnfree = true; };
      };
      pkg = pkgs.python3Packages.callPackage ./derivation.nix {};
      dev = import ./shell.nix { inherit pkg pkgs; };
    };
    packages = {
      py37-cpu = mkPkg false "python37";
      py37-gpu = mkPkg true "python37";
      py38-cpu = mkPkg false "python38";
      py38-gpu = mkPkg true "python38";
    };
  in {
    packages.x86_64-linux = builtins.mapAttrs (_: v: v.pkg) packages;
    overlays = builtins.mapAttrs (_: v: v.overlay) packages;
    defaultPackage.x86_64-linux = packages.py38-gpu.pkg;
    devShell.x86_64-linux = packages.py38-gpu.dev;
  };
}
