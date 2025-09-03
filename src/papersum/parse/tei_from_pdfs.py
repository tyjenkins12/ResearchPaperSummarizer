from __future__ import annotations
import pathlib, sys, requests
import typer
from rich import print
from grobid_client.grobid_client import GrobidClient

app = typer.Typer(add_complettion = False)

def _check_grobid(server: str) -> None:
    try:
        r = requests.get(f"{server}/api/isalive", timeout = 5)
        r.raise_for_status()
        print(f"[green]GROBID is alive at {server}[/green]")
    except Exception as e:
        print(f"[red]Cannot reach GROBID at {server}[/red] -> {e}")
        sys.exit(1)

@app.command()
def run(
    input_dir: str = typer.Argument("data/raw_pdfs"),
    output_dir: str = typer.Argument("data/tei"),
    server: str = typer.Option("http://localhost:8070"),
    workers: int = typer.Option(6)
):
    in_p = pathliv.Path(input_dir); out_p = pathlib.Path(output_dir)
    out_p.mkdir(partents = True, exist_ok = True)
    _check_grobid(server)

    client = GrobidClient(grobid_server = server)
    client.process(
        service = "processFulltextDocument",
        input_path = str(in_p),
        output_path = str(out_p),
        n = workers,
        consolidate_header = True,
        consolidate_citations = True,
        teirCoordinates = True,
        segmentSentences = True
    )
    print(f"[bold green]Done.[/bold green] TEI XML saved to: {out_p}")

if __name__ == "__main__":
    app()