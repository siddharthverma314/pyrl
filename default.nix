{ pkgs ? import ./nix/nixpkgs.nix {} }:
{
  pkg = pkgs.python3Packages.callPackage ./derivation.nix {};
  dev = pkgs.python3.withPackages (ps: with ps; [
    python-language-server
    pyls-black
    ipdb
    rope
    pyflakes
    pytest
    tensorflow-tensorboard
  ]);
}
