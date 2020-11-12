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
    pkgs_cpu_py38 = import ./nix/nixpkgs.nix { inherit inputs; cudaSupport = false; python = "python38"; };
    pkgs_gpu_py38 = import ./nix/nixpkgs.nix { inherit inputs; cudaSupport = true; python = "python38"; };
    pkgs_cpu_py37 = import ./nix/nixpkgs.nix { inherit inputs; cudaSupport = false; python = "python37"; };
    pkgs_gpu_py37 = import ./nix/nixpkgs.nix { inherit inputs; cudaSupport = true; python = "python37"; };
  in
    {
      pkgs_cpu_py38 = import default.nix { pkgs = pkgs_cpu_py38; };
      pkgs_gpu_py38 = import default.nix { pkgs = pkgs_gpu_py38; };
      pkgs_cpu_py37 = import default.nix { pkgs = pkgs_cpu_py37; };
      pkgs_gpu_py37 = import default.nix { pkgs = pkgs_gpu_py37; };
    }
}
