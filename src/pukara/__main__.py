"""Run the Pukara gateway: python -m pukara"""

import uvicorn

from pukara.config import load_config

config = load_config()
uvicorn.run("pukara.app:create_app", host=config.host, port=config.port, factory=True)
