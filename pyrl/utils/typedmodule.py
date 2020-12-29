from torch.nn import Module


class TypedModule(Module):
    DEBUG = False

    def __init__(self, input_spec, output_spec):
        super().__init__(self)
        self.input_spec = input_spec
        self.output_spec = output_spec

    def forward(self, inp):
        if self.DEBUG:
            assert self.input_spec.contains(inp)
        assert inp
