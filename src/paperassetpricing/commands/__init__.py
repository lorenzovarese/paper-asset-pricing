import typer

from .aggregate import aggregate as _agg
from .experiment import experiment as _exp
from .portfolio import run_portfolio as _port

app = typer.Typer(
    invoke_without_command=True,
    help="P.A.P.E.R. CLI: Platform for Asset Pricing Experiment & Research",
)


# If no command is given, show help.
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# Register them under the names you want
app.command("aggregate")(_agg)
app.command("experiment")(_exp)
app.command("portfolio")(_port)
