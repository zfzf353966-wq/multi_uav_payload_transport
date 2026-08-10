from setuptools import find_packages, setup

package_name = "logistics_planner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gbc",
    maintainer_email="gbc@example.com",
    description="Known-map task allocation and 3D planning baseline.",
    license="Apache-2.0",
    scripts=["scripts/traditional_baseline", "scripts/offboard_mission"],
)
