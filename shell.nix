let
  pkgs = import ./nix/nixpkgs.nix;
  pyrl = pkgs.python3Packages.callPackage ./default.nix {};
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      pyrl

      python-language-server
      pyls-black
      ipdb
      rope
      pyflakes
      pytest
    ]))
  ];
}
