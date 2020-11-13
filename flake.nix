{
  inputs = {
    cpprb = {
      url = github:ymd-h/cpprb;
      flake = false;
    };
    flatten-dict = {
      url = github:ianlini/flatten-dict;
      flake = false;
    };
    gitignore = {
      url = github:hercules-ci/gitignore;
      flake = false;
    };
    mujoco = {
      url = https://www.roboti.us/download/mjpro150_linux.zip;
      flake = false;
    };
    mujoco-py = {
      url = github:siddharthverma314/mujoco-py;
      flake = false;
    };
    pyGLFW = {
      url = github:FlorianRhiem/pyGLFW;
      flake = false;
    };
    scikit-video = {
      url = github:scikit-video/scikit-video;
      flake = false;
    };
    nixpkgs.url = github:NixOS/nixpkgs;
  };
  outputs = inputs: let
    mkPkg = cudaSupport: python: let
      pkgs = import ./nix/nixpkgs.nix {
        inherit inputs;
        cudaSupport = false;
        python = "python38";
      };
    in
      pkgs.python3Packages.callPackage ./derivation.nix {};

    pkgs_cpu_py37 = mkPkg false "python37";
    pkgs_gpu_py37 = mkPkg true "python37";
    pkgs_cpu_py38 = mkPkg false "python38";
    pkgs_gpu_py38 = mkPkg true "python38";
  in {
    packages.x86_64-linux = { inherit pkgs_cpu_py37 pkgs_cpu_py38 pkgs_gpu_py37 pkgs_gpu_py38; };
    defaultPackage.x86_64-linux = pkgs_gpu_py38;
  };
}
