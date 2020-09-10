let
  pkgs = import ./nix/nixpkgs.nix {};
  pyrl = pkgs.python3Packages.callPackage ./default.nix {};
in
pkgs.mkShell {
  buildInputs = [
    pyrl.pkg
    pyrl.dev
  ];
}
