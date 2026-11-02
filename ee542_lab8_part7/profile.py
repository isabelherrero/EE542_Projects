
import geni.portal as portal
import geni.rspec.pg as pg
import geni.rspec.emulab

pc = portal.Context()
request = pc.makeRequestRSpec()

lan = request.LAN("lan")

for index in range(2):
    node = request.RawPC("node{}".format(index))
    node.hardware_type = "d7525"
    node.disk_image = (
        "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"
    )

    interface = node.addInterface("if{}".format(index))
    interface.addAddress(
        pg.IPv4Address(
            "10.10.1.{}".format(index + 1),
            "255.255.255.0"
        )
    )
    lan.addInterface(interface)

    storage = node.Blockstore("docker-storage-{}".format(index), "/mydata")
    storage.size = "100GB"
    storage.placement = "nonsysvol"

pc.printRequestRSpec(request)
