#!/bin/bash
# Find all NVIDIA PCIe devices (both VGA controller and Audio controller)
devices=$(lspci -D | grep -i nvidia | awk '{print $1}')

if [ -z "$devices" ]; then
    echo "No NVIDIA devices found on the PCIe bus. It might already be detached."
    exit 0
fi

echo "Found NVIDIA devices to detach:"
echo "$devices"
echo

for dev in $devices; do
    if [ -e "/sys/bus/pci/devices/$dev/remove" ]; then
        echo "Removing device $dev..."
        echo 1 | sudo tee "/sys/bus/pci/devices/$dev/remove" > /dev/null
    else
        echo "Device $dev has no remove interface (or already removed)."
    fi
done

echo "NVIDIA eGPU has been safely detached from the PCIe bus!"
