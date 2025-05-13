from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import ensuredir
import os
import shutil

class ImageBlenderDirective(Directive):
    required_arguments = 1  # ID of the widget
    def parse_global_weight(value):
        try:
            val = float(value)
            if 0 <= val <= 1:
                return val
            raise ValueError
        except ValueError:
            raise ValueError(f"global_weight must be a float in [0,1], got '{value}'")

    option_spec = {
        "images": lambda x: [img.strip() for img in x.split(",")],
        "labels": lambda x: [lbl.strip() for lbl in x.split(";;")],
        "base": str,  # Image path as a string
        "global_weight": parse_global_weight,  # Float in [0,1]
    }

    def run(self):
        env = self.state.document.settings.env
        builder = env.app.builder
        widget_id = self.arguments[0]
        base = self.options.get("base")
        lam = self.options.get("global_weight")
        images = self.options.get("images", [])
        labels = self.options.get("labels", [])

        img_uris = []
        for src_img in images:
            rel_img_path, abs_img_path = env.relfn2path(src_img)
            if not os.path.exists(abs_img_path):
                return [nodes.error(None, nodes.paragraph(text=f"Missing image: {src_img}"))]

            env.note_dependency(abs_img_path)  # Track for rebuilds

            # Ensure image is copied to _images/
            images_dir = os.path.join(env.app.outdir, builder.imagedir)
            ensuredir(images_dir)
            dest_img_path = os.path.join(images_dir, os.path.basename(abs_img_path))
            if not os.path.exists(dest_img_path):
                shutil.copyfile(abs_img_path, dest_img_path)

            # Store correct image reference for HTML output
            img_uris.append(f'"../{builder.imagedir}/{os.path.basename(abs_img_path)}"')

        base_uri = None
        if base:
            rel_img_path, abs_img_path = env.relfn2path(base)
            if not os.path.exists(abs_img_path):
                return [nodes.error(None, nodes.paragraph(text=f"Missing image: {base}"))]

            env.note_dependency(abs_img_path)
            images_dir = os.path.join(env.app.outdir, builder.imagedir)
            ensuredir(images_dir)
            dest_img_path = os.path.join(images_dir, os.path.basename(abs_img_path))
            if not os.path.exists(dest_img_path):
                shutil.copyfile(abs_img_path, dest_img_path)
            base_uri = f'"../{builder.imagedir}/{os.path.basename(abs_img_path)}"'

        img_list = "[" + ", ".join(img_uris) + "]"
        labels_list = "[" + ", ".join(f'"{lbl}"' for lbl in labels) + "]" if labels else "null"
        base_str = base_uri if base_uri else "null"
        lam_str = lam if lam is not None else 0.5

        html = f"""
        <div id="{widget_id}" class="image-blender-container"></div>
        <script>
        createImageBlender({{
            id: "{widget_id}",
            imageSources: {img_list},
            labels: {labels_list},
            baseImage: {base_str},
            globalWeight: {lam_str}
        }});
        </script>
        """
        return [nodes.raw("", html, format="html")]

def setup(app):
    app.add_directive("blender", ImageBlenderDirective)
    app.add_js_file("js/blender.js")
    app.add_css_file("css/blender.css")