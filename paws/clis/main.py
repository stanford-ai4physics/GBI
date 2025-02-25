import click

__all__ = ['cli']

class NaturalOrderGroup(click.Group):
    """Command group trying to list subcommands in the order they were added.

    Make sure you initialize the `self.commands` with OrderedDict instance.

    With decorator, use::

        @click.group(cls=NaturalOrderGroup, commands=OrderedDict())
    """

    def list_commands(self, ctx):
        """List command names as they are in commands dict.

        If the dict is OrderedDict, it will preserve the order commands
        were added.
        """
        return self.commands.keys()

class DelimitedStr(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return [i.strip() for i in value.split(",")]
        except Exception:
            raise click.BadParameter(value)

@click.group(cls=NaturalOrderGroup)
def cli():
    pass
    