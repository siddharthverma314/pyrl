{ pkgs, pkg }:
pkgs.buildEnv {
  name = "pyrl-dev";
  paths = [
    pkgs.python-language-server
    (pkgs.python3.withPackages (ps: with ps; [
      pkg
      pyls-black
      ipdb
      rope
      pyflakes
      pytest
      tensorflow-tensorboard_2
    ]))
  ];
}
