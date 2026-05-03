import io
import pathlib
from csv import DictReader

from bson import ObjectId
from fasthtml.common import (A, Button, Details, Div, Li, Span, Summary, Ul,
                             UploadFile)
from monsterui.all import H1, H3, Theme, fast_app

from ..util.log import get_logger
from ..db.mongo import categories, rows

logger = get_logger(__name__)
features_app, rt = fast_app(hdrs=Theme.blue.headers())


@rt("/{category_id}/{feature_id}/{file_id}", methods=["GET"])
async def get_file(category_id: str, feature_id: str, file_id: str, page: int = 1, limit: int = 40):
    try:
        category_bson_id = ObjectId(category_id)
        feature_bson_id = ObjectId(feature_id)
        file_bson_id = ObjectId(file_id)
    except:
        return Div("Invalid file id", cls="text-red-500")

    category_doc = categories().find_one({"_id": category_bson_id})

    if not category_doc:
        return Div(f"No category with id {category_bson_id}", cls="text-red-500")

    existing_feature = next(
        (f for f in category_doc["features"] if f["_id"] == feature_bson_id), None)
    if not existing_feature:
        return Div(f"No feature with id {category_bson_id}", cls="text-red-500")

    skip = (page - 1) * limit

    cursor = (
        rows()
        .find({"file_id": file_bson_id})
        .sort([("anomaly.anomaly_score", 1), ("anomaly_score", 1), ("_id", 1)])
        .skip(skip)
        .limit(limit)
    )

    preview_rows = list(cursor)

    if not preview_rows:
        return Div("No rows found")

    total_rows = rows().count_documents({"file_id": file_bson_id})
    total_pages = (total_rows + limit - 1) // limit
    file_name = preview_rows[0]["file_name"]

    # ---- tile renderer ----
    def render_tile(row):
        fields = {**row["data"], **row["anomaly"]}

        normal_items = []
        skill_items = []
        misc_items = []
        log_items = []

        def format_value(v):
            if v is None:
                return v

            s = str(v)

            try:
                i = int(s)
                return f"{i:,}".replace(",", " ")
            except ValueError:
                pass

            try:
                f = float(s)
                return f"{f:.6f}"
            except ValueError:
                pass

            return s

        for k, v in fields.items():
            formatted_value = format_value(v)
            
            item = Div(
                Span(k, cls="font-semibold text-xs"),
                Span(formatted_value, cls="text-xs"),
                cls="flex justify-between"
            )

            if k.startswith("skills_"):
                skill_items.append(item)
            elif k.startswith("misc_"):
                if formatted_value != '0':
                    misc_items.append(item)
            elif k.startswith("log_"):
                log_items.append(item)
            elif k not in ('timestamp'):
                normal_items.append(item)

        skills_section = None
        misc_section = None
        log_section = None
        if skill_items:
            skills_section = Details(
                Summary("Skills", cls="cursor-pointer font-semibold text-xs"),
                Div(*skill_items, cls="space-y-1 mt-1"),
                cls="border-t pt-1 mt-1"
            )
        if misc_items:
            misc_section = Details(
                Summary("Misc", cls="cursor-pointer font-semibold text-xs"),
                Div(*misc_items, cls="space-y-1 mt-1"),
                cls="border-t pt-1 mt-1"
            )
        if log_items:
            log_section = Details(
                Summary("Log", cls="cursor-pointer font-semibold text-xs"),
                Div(*log_items, cls="space-y-1 mt-1"),
                cls="border-t pt-1 mt-1"
            )

        is_anomaly = row["anomaly"].get('is_anomaly', False)

        return Div(
            Div(
                f"Anomaly: {is_anomaly}",
                cls="font-bold mb-2"
            ),
            Div(
                *normal_items,
                cls="space-y-1"
            ),
            skills_section if skills_section else "",
            misc_section if misc_section else "",
            log_section if log_section else "",
            cls=f"""
                border rounded-lg p-3 shadow-sm
                {'border-green-500' if is_anomaly else 'border-red-500'}
                hover:shadow-md transition
                text-xs
            """
        )

    grid = Div(
        *[render_tile(r) for r in preview_rows],
        cls="""
            grid
            grid-cols-4
            gap-4
        """
    )

    pagination = Div(
        Button(
            "Prev",
            hx_get=f"/category/{category_id}/{feature_id}/{file_id}?page={page-1}&limit={limit}",
            hx_target="#file-grid",
            disabled=page <= 1
        ),
        Span(f"Page {page} / {total_pages}", cls="mx-4"),
        Button(
            "Next",
            hx_get=f"/category/{category_id}/{feature_id}/{file_id}?page={page+1}&limit={limit}",
            hx_target="#file-grid",
            disabled=page >= total_pages
        ),
        cls="flex items-center gap-2 mt-4"
    )

    return Div(
        H1(f"{category_doc['category_name']}: {existing_feature['feature_name']}",
           cls="font-bold mb-3"),
        H3(f"file: {file_name}", cls="font-bold mb-3"),
        grid,
        pagination,
        id="file-grid"
    )


@rt("/{category_id}/{feature_id}", methods=["GET"])
async def get_feature(category_id: str, feature_id: str):
    try:
        category_bson_id = ObjectId(category_id)
        feature_bson_id = ObjectId(feature_id)
    except:
        return Div("Invalid IDs", cls="text-red-500")

    category_doc = categories().find_one({"_id": category_bson_id})

    if not category_doc:
        return Div(f"No category with id {category_bson_id}", cls="text-red-500")

    existing_feature = next(
        (f for f in category_doc["features"] if f["_id"] == feature_bson_id), None)
    if not existing_feature:
        return Div(f"No feature with id {category_bson_id}", cls="text-red-500")

    file_list = Ul(
        *[
            Li(
                A(
                    file["file_name"],
                    hx_get=f"/category/{category_id}/{feature_id}/{file['_id']}",
                    hx_target="#main",
                )
            )
            for file in existing_feature["files"]
        ],
        cls="space-y-1"
    )

    return Div(
        H1(f"Name: {existing_feature["feature_name"]}",
            cls="font-bold mb-2"),
        Div(f"Category id: {category_id} with feature id: {feature_id}",
            cls="font-bold mb-2"),
        file_list
    )


@rt("/upload", methods=["POST"])
async def post(category_name: str, feature_name: str, csv_files: list[UploadFile]):
    category_name = category_name.strip().replace(" ", "_")
    feature_name = feature_name.strip().replace(" ", "_")
    
    if not feature_name:
        return {"error": "Feature name is empty"}
    
    if not category_name:
        return {"error": "Category name is empty"}

    if not csv_files or not len(csv_files):
        return {"error": "Please upload CSV files"}

    category_doc = categories().find_one({"category_name": category_name})

    if not category_doc:
        category_doc = {
            "category_name": category_name,
            "features": []
        }
        category_id = categories().insert_one(category_doc).inserted_id
        category_doc["_id"] = category_id
    else:
        category_id = category_doc["_id"]

    existing_feature = next(
        (f for f in category_doc["features"] if f["feature_name"] == feature_name), None)
    if existing_feature:
        return {"error": "Feature already exists in this category"}

    feature_doc = {
        "_id": ObjectId(),
        "feature_name": feature_name,
        "files": []
    }

    uploaded_files = []
    anomaly_stats = {}

    for csv_file in csv_files:
        if not csv_file.filename.endswith(".csv"):  # type: ignore
            continue

        csv_file_path = pathlib.Path(str(csv_file.filename))
        csv_file_name = csv_file_path.stem
        logger.debug(f'start: {csv_file_name}')

        text_file = io.TextIOWrapper(csv_file.file, encoding="utf-8")
        reader = list(DictReader(text_file))

        ranked = sorted(
            reader,
            key=lambda r: float(r.get("anomaly_score", 0)),
        )
        n = len(ranked)
        MAX_RANK = int(0.05 * n)

        file_id = ObjectId()
        docs = []

        for rank, row in enumerate(ranked, start=1):

            is_anomaly = row.get("is_anomaly", "false").lower() == "true"
            sani_row = {k: v for k, v in row.items() if k not in (
                "anomaly_label", "anomaly_score", "is_anomaly")}
            # inverse_score = 1 / rank if rank <= MAX_RANK else 0.0
            inverse_score = (n - rank + 1) / n if rank <= MAX_RANK else 0

            docs.append({
                "category_id": category_id,
                "feature_id": feature_doc["_id"],
                "file_id": file_id,
                "file_name": csv_file_name,
                "anomaly": {
                    "is_anomaly": is_anomaly,
                    "anomaly_label": int(row.get("anomaly_label", 0)),
                    "anomaly_score": float(row.get("anomaly_score", 0)),
                },
                "data": sani_row
            })
            username = row.get("username")
            if username and is_anomaly:
                if username not in anomaly_stats:
                    anomaly_stats[username] = {
                        "count": 0,
                        "inverse_score": 0,
                        "data": sani_row,
                    }

                anomaly_stats[username]["count"] += 1
                anomaly_stats[username]["inverse_score"] += inverse_score

            if len(docs) >= 1000:
                rows().insert_many(docs)
                docs = []

        if docs:
            rows().insert_many(docs)

        file_ref = {
            "_id": file_id,
            "file_name": csv_file_name,
            "rows_count": len(reader)
        }

        feature_doc["files"].append(file_ref)
        uploaded_files.append(csv_file_name)

    if anomaly_stats:
        agg_file_id = ObjectId()
        agg_docs = []
        for username, stats in anomaly_stats.items():
            agg_docs.append({
                "category_id": category_id,
                "feature_id": feature_doc["_id"],
                "file_id": agg_file_id,
                "file_name": "anomaly_counts",
                "anomaly": {
                    "is_anomaly": True,
                    "anomaly_count": stats["count"],
                    "anomaly_label": -stats["count"],
                    "anomaly_score": -stats["inverse_score"],
                },
                "data": stats["data"],
            })

            if len(agg_docs) >= 1000:
                rows().insert_many(agg_docs)
                agg_docs = []

        if agg_docs:
            rows().insert_many(agg_docs)

        feature_doc["files"].append({
            "_id": agg_file_id,
            "file_name": "anomaly_counts",
            "rows_count": len(anomaly_stats)
        })

    categories().update_one(
        {"_id": category_id},
        {"$push": {"features": feature_doc}}
    )

    return {
        "status": "ok",
        "category": category_name,
        "feature": feature_name,
        "file_count": len(uploaded_files),
        "files_uploaded": uploaded_files
    }
