from fasthtml.common import A, Div, Li, Optgroup, Option, Select, Ul, fast_app
from monsterui.all import (H1, Button, ButtonT, Container, ContainerT, Form,
                           Input, Theme, fast_app)

from ..util.common import HSType
from ..db.mongo import categories

index_app, rt = fast_app(hdrs=Theme.blue.headers())


@rt("/")
async def index():
    category_docs = categories().find().sort("uploaded_at", -1)

    category_list = Ul(
        *[
            Li(
                Div(
                    cat["category_name"],
                    cls="font-semibold cursor-pointer py-2 px-2.5 hover:bg-sidebar-nav-hover rounded-lg"
                ),
                Ul(
                    *[
                        Li(
                            A(
                                feature["feature_name"],
                                hx_get=f"/category/{cat['_id']}/{feature['_id']}",
                                hx_target="#main",
                                cls="flex items-center gap-x-3.5 py-1 px-4 text-sm text-sidebar-nav-foreground rounded-lg hover:bg-sidebar-nav-hover focus:outline-hidden focus:bg-sidebar-nav-focus"
                            )
                        )
                        for feature in cat.get("features", [])
                    ],
                    cls="pl-4 space-y-1"
                ),
                cls="mb-1"
            )
            for cat in category_docs
        ],
        cls="space-y-2"
    )

    skill_options = [Option(A(item.name))
                     for item in HSType if item.is_skill()]
    misc_options = [Option(A(item.name)) for item in HSType if item.is_misc()]

    category_select = Select(
        Optgroup(*skill_options, label="Skills"),
        Optgroup(*misc_options, label="Miscs"),
        name="category_name",
        cls='mb-2 flex items-center'
    )

    upload_form = Form(
        Container(
            category_select,
            Input(name="feature_name", placeholder="Feature name", cls="mb-2"),
            Input(type="file", name="csv_files", accept=".csv",
                  webkitdirectory=True, multiple=True, cls="mb-2"),
            Button("Upload", cls=ButtonT.primary),
            cls=ContainerT.sm
        ),
        hx_post="/category/upload",
        hx_target="#results",
        hx_encoding="multipart/form-data"
    )

    main_content = Div(
        upload_form,
        Div(id="results"),
        id="main",
        cls="flex-1 p-4"
    )

    return Container(
        A(
            H1("OSRS HS ML Platform", cls="cursor-pointer"),
            href="/",
            cls="no-underline text-layer-foreground"
        ),
        Div(
            Div(
                category_list,
                cls="w-64 flex-shrink-0 h-screen sticky top-0 overflow-y-auto bg-sidebar border-e border-sidebar-line"
            ),
            Div(
                main_content,
                cls="flex-1 overflow-y-auto"
            ),
            cls="flex"
        )
    )
