import os
import glob
import argparse
import numpy as np
import pandas as pd

class InstanceType:
    def __init__(
            self,
            cost_per_hour=None,
            cpu_count=None,
            cpu_model=None,
            ram=None,
            gpu_count=None,
            gpu_model=None,
            instance_class=None):
        self.cost_per_hour = cost_per_hour
        self.cpu_count = cpu_count
        self.cpu_model = cpu_model
        self.ram = ram
        self.gpu_count = gpu_count
        self.gpu_model = gpu_model
        self.instance_class = instance_class

class LogType:
    def __init__(self):
        self.images = []
        self.model = []
        self.instance = []
        self.instance_class = []
        self.epoch_num = []
        self.time_per_epoch = []
        self.cost_per_hour = []
        self.cpu_count = []
        self.cpu_model = []
        self.ram = []
        self.gpu_count = []
        self.gpu_model = []
        self.cost_per_epoch = []
        self.total_time = []
        self.total_cost = []
        self.batch_size = []
        self.model_size = []
        self.gpu_mem_allocated = []
        self.percent_gpu_util = []

instances = {
    "g4dn-xl": InstanceType(
        cost_per_hour=0.526,
        cpu_count=4,
        cpu_model="Cascade Lake P-8259L",
        ram=16,
        gpu_count=1,
        gpu_model="t4",
        instance_class="g4"),
    "g4dn-2xl": InstanceType(
        cost_per_hour=0.752,
        cpu_count=8,
        cpu_model="Cascade Lake P-8259L",
        ram=32,
        gpu_count=1,
        gpu_model="t4",
        instance_class="g4"),
    "g4dn-4xl": InstanceType(
        cost_per_hour=1.204,
        cpu_count=16,
        cpu_model="Cascade Lake P-8259L",
        ram=64,
        gpu_count=1,
        gpu_model="t4",
        instance_class="g4"),
    "g4dn-8xl": InstanceType(
        cost_per_hour=2.176,
        cpu_count=32,
        cpu_model="Cascade Lake P-8259L",
        ram=128,
        gpu_count=1,
        gpu_model="t4",
        instance_class="g4"),
    "g4dn-12xl": InstanceType(
        cost_per_hour=3.912,
        cpu_count=48,
        cpu_model="Cascade Lake P-8259L",
        ram=192,
        gpu_count=4,
        gpu_model="t4",
        instance_class="g4"),
    "g4dn-16xl": InstanceType(
        cost_per_hour=4.352,
        cpu_count=64,
        cpu_model="Cascade Lake P-8259L",
        ram=256,
        gpu_count=1,
        gpu_model="t4",
        instance_class="g4"),
    "p2-xl": InstanceType(
        cost_per_hour=0.900,
        cpu_count=4,
        cpu_model="Broadwell E5-2686 v4",
        ram=61,
        gpu_count=1,
        gpu_model="k80",
        instance_class="p2"),
    "p2-8xl": InstanceType(
        cost_per_hour=7.200,
        cpu_count=32,
        cpu_model="Broadwell E5-2686 v4",
        ram=488,
        gpu_count=8,
        gpu_model="k80",
        instance_class="p2"),
    "p3-2xl": InstanceType(
        cost_per_hour=3.060,
        cpu_count=8,
        cpu_model="Broadwell E5-2686 v4",
        ram=61,
        gpu_count=1,
        gpu_model="v100",
        instance_class="p3"),
    "p3-8xl": InstanceType(
        cost_per_hour=12.240,
        cpu_count=32,
        cpu_model="Broadwell E5-2686 v4",
        ram=244,
        gpu_count=4,
        gpu_model="v100",
        instance_class="p3"),
    "p3-16xl": InstanceType(
        cost_per_hour=24.480,
        cpu_count=64,
        cpu_model="Broadwell E5-2686 v4",
        ram=488,
        gpu_count=8,
        gpu_model="v100",
        instance_class="p3"),
    "p3dn-24xl": InstanceType(
        cost_per_hour=31.212,
        cpu_count=96,
        cpu_model="Skylake 8175",
        ram=768,
        gpu_count=8,
        gpu_model="v100",
        instance_class="p3"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="Log directory")
    args = parser.parse_args()

    epoch_df = pd.DataFrame()
    total_df = pd.DataFrame()

    epochs = LogType()
    totals = LogType()

    log_list = glob.glob(os.path.join(args.dir, "*.txt"))
    for log in log_list:
        log_split = log.split("/")[-1].split("_")
        images = int(log_split[1].replace("k", "000"))
        model = log_split[2]
        instance = f"{log_split[3]}-{log_split[4]}"
        instance_class = log_split[3]
        batch_size = None
        with open(log, "rt") as file:
            for line in file:
                if line.startswith("GPU memory allocated"):
                    gpu_mem_allocated = int(line[22:])

                if line.startswith("Model size"):
                    model_size = int(line[12:])

                if line.startswith("Batch size"):
                    batch_size = int(line[12:])

                if line.startswith("Epoch"):
                    epoch = line[6]
                    time_vals = [float(val) for val in line[9:].split(":")]
                    time = sum([a * b for a, b in zip(time_vals, [3600, 60, 1])])
                    epochs.images.append(images)
                    epochs.model.append(model)
                    epochs.instance.append(instance)
                    epochs.instance_class.append(instances[instance].instance_class)
                    epochs.epoch_num.append(epoch)
                    epochs.time_per_epoch.append(time)
                    epochs.cost_per_hour.append(instances[instance].cost_per_hour)
                    epochs.cpu_count.append(instances[instance].cpu_count)
                    epochs.cpu_model.append(instances[instance].cpu_model)
                    epochs.ram.append(instances[instance].ram)
                    epochs.gpu_count.append(instances[instance].gpu_count)
                    epochs.gpu_model.append(instances[instance].gpu_model)
                    epochs.cost_per_epoch.append(instances[instance].cost_per_hour/(3600) * time)
                    epochs.batch_size.append(batch_size)
                    epochs.model_size.append(model_size)
                    epochs.gpu_mem_allocated.append(gpu_mem_allocated)
                    epochs.percent_gpu_util.append((batch_size * model_size)/gpu_mem_allocated)

                if line.startswith("Total elapsed"):
                    time_vals = [float(val) for val in line.split()[-1].split(":")]
                    if "day" in line:
                        days = int(line.split()[2])
                        time_vals[0] += days * 24
                    time = sum([a * b for a, b in zip(time_vals, [3600, 60, 1])])
                    totals.images.append(images)
                    totals.model.append(model)
                    totals.instance.append(instance)
                    totals.instance_class.append(instances[instance].instance_class)
                    totals.total_time.append(time)
                    totals.cost_per_hour.append(instances[instance].cost_per_hour)
                    totals.cpu_count.append(instances[instance].cpu_count)
                    totals.cpu_model.append(instances[instance].cpu_model)
                    totals.ram.append(instances[instance].ram)
                    totals.gpu_count.append(instances[instance].gpu_count)
                    totals.gpu_model.append(instances[instance].gpu_model)
                    totals.total_cost.append(instances[instance].cost_per_hour/(3600) * time)
                    totals.batch_size.append(batch_size)
                    totals.model_size.append(model_size)
                    totals.gpu_mem_allocated.append(gpu_mem_allocated)
                    totals.percent_gpu_util.append((batch_size * model_size)/gpu_mem_allocated)

    total_df["images"] = totals.images
    total_df["model"] = totals.model
    total_df["instance"] = totals.instance
    total_df["instance_class"] = totals.instance_class
    total_df["total_time"] = totals.total_time
    total_df["cost_per_hour"] = totals.cost_per_hour
    total_df["cpu_count"] = totals.cpu_count
    total_df["cpu_model"] = totals.cpu_model
    total_df["ram"] = totals.ram
    total_df["gpu_count"] = totals.gpu_count
    total_df["gpu_model"] = totals.gpu_model
    total_df["cost_per_run"] = totals.total_cost
    total_df["batch_size"] = totals.batch_size
    total_df["model_size"] = totals.model_size
    total_df["gpu_mem_allocated"] = totals.gpu_mem_allocated
    total_df["percent_gpu_util"] = totals.percent_gpu_util

    epoch_df["images"] = epochs.images
    epoch_df["model"] = epochs.model
    epoch_df["instance"] = epochs.instance
    epoch_df["instance_class"] = epochs.instance_class
    epoch_df["epoch"] = epochs.epoch_num
    epoch_df["time_per_epoch"] = epochs.time_per_epoch
    epoch_df["cost_per_hour"] = epochs.cost_per_hour
    epoch_df["cpu_count"] = epochs.cpu_count
    epoch_df["cpu_model"] = epochs.cpu_model
    epoch_df["ram"] = epochs.ram
    epoch_df["gpu_count"] = epochs.gpu_count
    epoch_df["gpu_model"] = epochs.gpu_model
    epoch_df["cost_per_epoch"] = epochs.cost_per_epoch
    epoch_df["batch_size"] = epochs.batch_size
    epoch_df["model_size"] = epochs.model_size
    epoch_df["gpu_mem_allocated"] = epochs.gpu_mem_allocated
    epoch_df["percent_gpu_util"] = epochs.percent_gpu_util

    for image in epoch_df["images"].unique():
        for instance in epoch_df["instance"].unique():
            costs = epoch_df.loc[(epoch_df["images"] == image) & (epoch_df["instance"] == instance), "cost_per_epoch"]
            times = epoch_df.loc[(epoch_df["images"] == image) & (epoch_df["instance"] == instance), "time_per_epoch"]

            total_df.loc[(total_df["images"] == image) & (total_df["instance"] == instance), "mean_epoch_time"] = times.mean()
            total_df.loc[(total_df["images"] == image) & (total_df["instance"] == instance), "std_epoch_time"] = times.std()
            total_df.loc[(total_df["images"] == image) & (total_df["instance"] == instance), "mean_epoch_cost"] = costs.mean()
            total_df.loc[(total_df["images"] == image) & (total_df["instance"] == instance), "std_epoch_cost"] = costs.std()

    total_df = total_df.sort_values(by=["instance", "images"])
    total_df = total_df.reset_index(drop=True)

    epoch_df = epoch_df.sort_values(by=["instance", "images"])
    epoch_df = epoch_df.reset_index(drop=True)

    epoch_df.to_csv("epoch_times.csv", index=False)
    total_df.to_csv("total_times.csv", index=False)
