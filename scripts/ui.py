from fasthtml.common import Mount, fast_app, serve
from monsterui.all import Theme, fast_app

from osrs_anomaly_ml.routes.features import features_app
from osrs_anomaly_ml.routes.index import index_app
from dotenv import load_dotenv
load_dotenv()

routes = [
    Mount("/category", features_app, name="categories"),
    Mount("/", index_app, name="index"),
]

app, rt = fast_app(live=True, hdrs=Theme.blue.headers(), routes=routes)


if __name__ == '__main__':
    serve()
