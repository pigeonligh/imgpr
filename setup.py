import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="imgpr",
	version="0.0.1",
	author="lightning",
	author_email="cx24321@hotmail.com",
	description="A Simple Image Process Library (for Digital Image Process Lesson)",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://git.pigeonligh.com/pigeonligh/image_process",
	packages=setuptools.find_packages(),
	entry_points={},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	]
)
