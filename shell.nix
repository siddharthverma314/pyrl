{ pkgs, pkg }:
pkgs.mkShell {
  buildInputs = [
    pkgs.nodePackages.pyright
    (pkgs.python3.withPackages (ps: with ps; [
      black
      pkg
      ipdb
      rope
      pyflakes
      pytest
      tensorflow-tensorboard_2
    ]))
  ];
}
