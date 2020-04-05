import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="imgpr",
	version="0.0.2",
	author="pigeonligh",
	author_email="pigeonligh@hotmail.com",
	description="A Simple Image Process Library (for Digital Image Process Lesson)",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/pigeonligh/imgpr",
	packages=setuptools.find_packages(),
	entry_points={},
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	]
)
