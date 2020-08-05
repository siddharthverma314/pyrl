let
  pkgs = import ./nix/nixpkgs.nix;
in
pkgs.python3Packages.buildPythonPackage {
  pname = "pyrl";
  version = "0.1.0";

  src = ./.;

  propagatedBuildInputs = (with pkgs.python3Packages; [
    pytorch
    gym
    transformers
    cpprb
    flatten-dict
    termcolor
    pygments
    tabulate
  ]);
}
