import subprocess

def main_cli():
	print("Installing freeimage binaries for imageio...")
	subprocess.run(["imageio_download_bin", "freeimage"])
	print("All done.")

if __name__ == '__main__':
    main_cli()