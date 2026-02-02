from flask import Flask
from pathlib import Path

def create_app():
    base_dir = Path(__file__).resolve().parent.parent

    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    app.config["SECRET_KEY"] = "dev"

    from .routes import bp
    app.register_blueprint(bp)

    return app
