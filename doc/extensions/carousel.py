from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util import ensuredir
import os
import shutil

class ImageCarouselDirective(Directive):
    has_content = False
    required_arguments = 1  # Image source
    optional_arguments = 0
    option_spec = {
        "id": directives.unchanged_required,
        "width": directives.nonnegative_int,
        "height": directives.nonnegative_int,
        "frames": directives.nonnegative_int,
        "hoffset": directives.nonnegative_int,
        "voffset": directives.nonnegative_int,
        "padleft": directives.nonnegative_int,
        "padright": directives.nonnegative_int,
        "direction": lambda x: directives.choice(x, ('H', 'V')),
        "labels": directives.unchanged,  # Labels separated by ;;
    }

    def run(self):
        env = self.state.document.settings.env
        builder = env.app.builder
        src_img = self.arguments[0]

        # Resolve absolute source path
        abs_src_path = os.path.join(env.app.srcdir, src_img)
        rel_img_path, abs_img_path = env.relfn2path(src_img)
        if not os.path.exists(abs_img_path):
            return [nodes.error(None, nodes.paragraph(text=f"Missing image: {src_img}"))]

        # Track for rebuilds
        env.note_dependency(abs_img_path)

        # Ensure image is copied to _images/
        images_dir = os.path.join(env.app.outdir, builder.imagedir)
        ensuredir(images_dir)
        dest_img_path = os.path.join(images_dir, os.path.basename(abs_img_path))
        if not os.path.exists(dest_img_path):
            shutil.copyfile(abs_img_path, dest_img_path)

        # Use correct reference for Sphinx HTML output
        img_uri = f"../{builder.imagedir}/{os.path.basename(abs_img_path)}"

        # Parse options
        options = {k: v for k, v in self.options.items()}
        labels = options.get("labels", "").split(";;") if "labels" in options else []
        labels_js = "[" + ",".join(f'"{label.strip()}"' for label in labels) + "]"

        # Generate the HTML
        div_id = options["id"]
        js_init = f"""
        <div id="{div_id}"></div>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                createSlideViewer({{
                    id: "{div_id}",
                    imageSrc: "{img_uri}",
                    slideWidth: {options.get("width", 512)},
                    slideHeight: {options.get("height", 512)},
                    hOffset: {options.get("hoffset", 0)},
                    vOffset: {options.get("voffset", 0)},
                    frames: {options.get("frames", 10)},
                    direction: "{options.get("direction", "H")}",
                    padLeft: {options.get("padleft", 0)},
                    padRight: {options.get("padright", 0)},
                    labels: {labels_js}
                }});
            }});
        </script>
        """

        return [nodes.raw("", js_init, format="html")]

def setup(app):
    app.add_directive("carousel", ImageCarouselDirective)
    app.add_js_file("js/carousel.js")
    app.add_css_file("css/carousel.css")
