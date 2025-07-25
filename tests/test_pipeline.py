import pandas as pd
from flowserv import Pipeline
from flowserv.steps import Load, Model, Serve


def test_pipeline(tmp_path):
    csv = tmp_path / "data.csv"
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    df.to_csv(csv, index=False)

    pipeline = Pipeline([Load(str(csv)), Model(target="target")])
    model = pipeline.execute()
    assert hasattr(model, "predict")

    serve = Serve()
    app = serve.run()
    routes = [route.path for route in app.router.routes]
    assert "/health" in routes
    assert "/model/predict" in routes
