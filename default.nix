{ pkgs ? import ./nix/nixpkgs.nix {} }:
pkgs.python3Packages.callPackage ./derivation.nix {}
