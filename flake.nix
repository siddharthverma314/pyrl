{
  inputs = {
    cpprb = { url = github:ymd-h/cpprb; flake = false; };
    flatten-dict = { url = github:ianlini/flatten-dict; flake = false; };
    gitignore = { url = github:hercules-ci/gitignore; flake = false; };
    mujoco = { url = https://www.roboti.us/download/mjpro150_linux.zip; flake = false; };
    mujoco-py = { url = github:siddharthverma314/mujoco-py; flake = false; };
    pyGLFW = { url = github:FlorianRhiem/pyGLFW; flake = false; };
    scikit-video = { url = github:scikit-video/scikit-video; flake = false; };
    nixpkgs.url = github:NixOS/nixpkgs;
  };
  outputs = inputs: let
    mkPkg = cudaSupport: python: let
      pkgs = import ./nix/nixpkgs.nix {
        inherit inputs;
        cudaSupport = false;
        python = "python38";
      };
      pkg = pkgs.python3Packages.callPackage ./derivation.nix {};
    in { inherit pkg; dev = import ./shell.nix {inherit pkg pkgs; }; };
    packages = {
      py37_cpu = mkPkg false "python37";
      py37_gpu = mkPkg true "python37";
      py38_cpu = mkPkg false "python38";
      py38_gpu = mkPkg true "python38";
    };
  in {
    packages.x86_64-linux = builtins.mapAttrs (_: v: v.pkg) packages;
    defaultPackage.x86_64-linux = packages.py38_gpu.pkg;
    devShell.x86_64-linux = packages.py38_gpu.dev;
    nixpkgs = inputs.nixpkgs;
  };
}
