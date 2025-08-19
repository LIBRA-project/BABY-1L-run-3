import pathsim
from pathsim import Simulation, Connection
import numpy as np
import matplotlib.pyplot as plt
import pathview

from tritium_model import (
    neutron_rate,
    total_irradiation_time,
    k_wall,
    k_top,
    baby_model,
    measured_TBR,
    replacement_times_top,
    replacement_times_walls,
    gas_switch_deltatime,
)

from libra_toolbox.tritium.model import ureg, activity_to_quantity

# Create global variables
A_OV = (baby_model.A_wall.to("m^2")).magnitude
A_IV = (baby_model.A_top.to("m^2")).magnitude
baby_vol = (baby_model.volume.to("m^3")).magnitude
k_IV = k_top.to("m/s").magnitude
k_OV = k_wall.to("m/s").magnitude
neutron_rate = neutron_rate.to("n/s").magnitude
irradiation_time = total_irradiation_time.to("s").magnitude

S_IV = 1
S_OV = 1
k1 = 3e-17
k2 = 0.4e-4
# f_stick_IV = 0.15
f_stick_IV = 0.15
f_stick_OV = 0.9

initial_inv_wall_IV = activity_to_quantity(4 * ureg.Bq).to(ureg.particle).magnitude
initial_inv_wall_OV = 8e9
t_h2 = gas_switch_deltatime.to("s").magnitude

tbr = measured_TBR.to("dimensionless").magnitude

IV_gas_residence_time = 0.1 * baby_vol / (A_IV * k_IV)
OV_gas_residence_time = 2.2 * baby_vol / (A_OV * k_OV)

baby_residence_time = baby_vol / (k_IV * A_IV + k_OV * A_OV)

collection_efficiency = 0.95
conversion_efficiency = 1

# Create blocks
blocks, events = [], []

tbr_7 = pathsim.blocks.amplifier.Amplifier(gain=3e-3)
blocks.append(tbr_7)

baby_8 = pathview.custom_pathsim_blocks.Process(
    residence_time=baby_vol / (k_IV * A_IV + k_OV * A_OV),
)
blocks.append(baby_8)

iv_vial_activity_21 = pathsim.blocks.scope.Scope(
    labels=[
        "IV bubbler (vial1)",
        "IV bubbler (vial2)",
        "IV bubbler (vial3)",
        "IV bubbler (vial4)",
    ]
)
blocks.append(iv_vial_activity_21)

soluble_vs_insoluble_1 = pathview.custom_pathsim_blocks.Splitter2(f1=0.01, f2=0.99)
blocks.append(soluble_vs_insoluble_1)

iv_bubbler_23 = pathview.custom_pathsim_blocks.Bubbler(
    conversion_efficiency=0.95,
    vial_efficiency=0.9,
    replacement_times=replacement_times_top.to("s").magnitude,
)
events_iv_bubbler_23 = iv_bubbler_23.create_reset_events()
events += events_iv_bubbler_23
blocks.append(iv_bubbler_23)

iv_vs_ov_24 = pathview.custom_pathsim_blocks.Splitter2(
    f1=A_IV * k_IV / (A_IV * k_IV + A_OV * k_OV),
    f2=A_OV * k_OV / (A_IV * k_IV + A_OV * k_OV),
)
blocks.append(iv_vs_ov_24)

soluble_vs_insoluble_25 = pathview.custom_pathsim_blocks.Splitter2(f1=0.01, f2=0.99)
blocks.append(soluble_vs_insoluble_25)

ov_bubbler_26 = pathview.custom_pathsim_blocks.Bubbler(
    conversion_efficiency=0.95,
    vial_efficiency=0.9,
    replacement_times=replacement_times_walls.to("s").magnitude,
)
events_ov_bubbler_26 = ov_bubbler_26.create_reset_events()
events += events_ov_bubbler_26
blocks.append(ov_bubbler_26)

environment_27 = pathview.custom_pathsim_blocks.Integrator()
blocks.append(environment_27)

ov_vial_activity_28 = pathsim.blocks.scope.Scope(
    labels=[
        "OV bubbler (vial1)",
        "OV bubbler (vial2)",
        "OV bubbler (vial3)",
        "OV bubbler (vial4)",
    ]
)
blocks.append(ov_vial_activity_28)

baby_inventory_30 = pathsim.blocks.scope.Scope(labels=["BABY (inv)"])
blocks.append(baby_inventory_30)

stepsource_31 = pathsim.blocks.sources.StepSource(
    amplitude=neutron_rate, tau=irradiation_time
)
blocks.append(stepsource_31)

stepsource_32 = pathsim.blocks.sources.StepSource(amplitude=neutron_rate, tau=0)
blocks.append(stepsource_32)

neutron_source_33 = pathsim.blocks.scope.Scope(labels=["addsub 57"])
blocks.append(neutron_source_33)

cumulative_release_34 = pathsim.blocks.scope.Scope(labels=["IV", "OV"])
blocks.append(cumulative_release_34)

iv_35 = pathview.custom_pathsim_blocks.Integrator()
blocks.append(iv_35)

ov_36 = pathview.custom_pathsim_blocks.Integrator()
blocks.append(ov_36)

iv_gas_37 = pathview.custom_pathsim_blocks.Process(
    residence_time=0.1 * baby_vol / (A_IV * k_IV),
)
blocks.append(iv_gas_37)

ov_gas_38 = pathview.custom_pathsim_blocks.Process(
    residence_time=4 * baby_vol / (A_OV * k_OV),
)
blocks.append(ov_gas_38)

wall_sticking_39 = pathview.custom_pathsim_blocks.Splitter2(
    f1=f_stick_IV, f2=1 - f_stick_IV
)
blocks.append(wall_sticking_39)

wall_sticking_40 = pathview.custom_pathsim_blocks.Splitter2(
    f1=f_stick_OV, f2=1 - f_stick_OV
)
blocks.append(wall_sticking_40)

iv_wall_42 = pathview.custom_pathsim_blocks.Integrator(
    initial_value=initial_inv_wall_IV,
)
blocks.append(iv_wall_42)

h2_43 = pathsim.blocks.sources.StepSource(tau=t_h2)
blocks.append(h2_43)

wall_reactions_45 = pathsim.blocks.function.Function(
    func=lambda c_h2, I: (k1 * (I / S_IV) ** 2, k2 * I / S_IV * c_h2)
)
blocks.append(wall_reactions_45)

wall_reactions_50 = pathsim.blocks.function.Function(
    func=lambda c_h2, I: (k1 * (I / S_OV) ** 2, k2 * I / S_OV * c_h2)
)
blocks.append(wall_reactions_50)

walls_53 = pathsim.blocks.scope.Scope(labels=["IV wall", "OV_walls"])
blocks.append(walls_53)

ov_walls_56 = pathview.custom_pathsim_blocks.Integrator(
    initial_value=initial_inv_wall_OV,
)
blocks.append(ov_walls_56)

addsub_57_57 = pathsim.blocks.adder.Adder(operations="+-")
blocks.append(addsub_57_57)

addsub_58_58 = pathsim.blocks.adder.Adder(operations="+--")
blocks.append(addsub_58_58)

addsub_59_59 = pathsim.blocks.adder.Adder(operations="+--")
blocks.append(addsub_59_59)


# Create events


# Create connections

connections = [
    Connection(tbr_7[0], baby_8[0]),
    Connection(soluble_vs_insoluble_1["source1"], iv_bubbler_23["sample_in_soluble"]),
    Connection(soluble_vs_insoluble_1["source2"], iv_bubbler_23["sample_in_insoluble"]),
    Connection(iv_bubbler_23["vial1"], iv_vial_activity_21[0]),
    Connection(iv_bubbler_23["vial2"], iv_vial_activity_21[1]),
    Connection(iv_bubbler_23["vial3"], iv_vial_activity_21[2]),
    Connection(iv_bubbler_23["vial4"], iv_vial_activity_21[3]),
    Connection(baby_8["mass_flow_rate"], iv_vs_ov_24[0]),
    Connection(soluble_vs_insoluble_25["source1"], ov_bubbler_26["sample_in_soluble"]),
    Connection(
        soluble_vs_insoluble_25["source2"], ov_bubbler_26["sample_in_insoluble"]
    ),
    Connection(iv_bubbler_23["sample_out"], environment_27[0]),
    Connection(ov_bubbler_26["sample_out"], environment_27[1]),
    Connection(ov_bubbler_26["vial1"], ov_vial_activity_28[0]),
    Connection(ov_bubbler_26["vial2"], ov_vial_activity_28[1]),
    Connection(ov_bubbler_26["vial3"], ov_vial_activity_28[2]),
    Connection(ov_bubbler_26["vial4"], ov_vial_activity_28[3]),
    Connection(baby_8["inv"], baby_inventory_30[0]),
    Connection(iv_35[0], cumulative_release_34[0]),
    Connection(ov_36[0], cumulative_release_34[1]),
    Connection(iv_gas_37["mass_flow_rate"], iv_35[0]),
    Connection(iv_gas_37["mass_flow_rate"], soluble_vs_insoluble_1[0]),
    Connection(ov_gas_38["mass_flow_rate"], ov_36[0]),
    Connection(ov_gas_38["mass_flow_rate"], soluble_vs_insoluble_25[0]),
    Connection(iv_vs_ov_24["source1"], wall_sticking_39[0]),
    Connection(iv_vs_ov_24["source2"], wall_sticking_40[0]),
    Connection(wall_sticking_40["source2"], ov_gas_38[0]),
    Connection(h2_43[0], wall_reactions_45[0]),
    Connection(iv_wall_42[0], wall_reactions_45[1]),
    Connection(wall_reactions_45[1], iv_gas_37[1]),
    Connection(wall_reactions_45[0], iv_gas_37[2]),
    Connection(wall_sticking_39["source2"], iv_gas_37[0]),
    Connection(wall_reactions_50[1], ov_gas_38[1]),
    Connection(iv_wall_42[0], walls_53[0]),
    Connection(wall_reactions_50[0], ov_gas_38[2]),
    Connection(h2_43[0], wall_reactions_50[0]),
    Connection(ov_walls_56[0], wall_reactions_50[1]),
    Connection(ov_walls_56[0], walls_53[1]),
    Connection(stepsource_31[0], addsub_57_57[1]),
    Connection(stepsource_32[0], addsub_57_57[0]),
    Connection(addsub_57_57[0], tbr_7[0]),
    Connection(addsub_57_57[0], neutron_source_33[0]),
    Connection(wall_sticking_40["source1"], addsub_58_58[0]),
    Connection(wall_reactions_50[0], addsub_58_58[1]),
    Connection(wall_reactions_50[1], addsub_58_58[2]),
    Connection(addsub_58_58[0], ov_walls_56[0]),
    Connection(wall_sticking_39["source1"], addsub_59_59[0]),
    Connection(wall_reactions_45[0], addsub_59_59[1]),
    Connection(wall_reactions_45[1], addsub_59_59[2]),
    Connection(addsub_59_59[0], iv_wall_42[0]),
]

# Create simulation
my_simulation = Simulation(
    blocks,
    connections,
    events=events,
    Solver=pathsim.solvers.SSPRK22,
    dt=100,
    dt_max=1.0,
    dt_min=1e-6,
    iterations_max=100,
    log=True,
    tolerance_fpi=1e-6,
    **{"tolerance_lte_rel": 1e-4, "tolerance_lte_abs": 1e-9},
)

if __name__ == "__main__":
    my_simulation.run(8 * 24 * 3600)

    # Optional: Plotting results
    scopes = [block for block in blocks if isinstance(block, pathsim.blocks.Scope)]
    fig, axs = plt.subplots(
        nrows=len(scopes), sharex=True, figsize=(10, 5 * len(scopes))
    )
    for i, scope in enumerate(scopes):
        plt.sca(axs[i] if len(scopes) > 1 else axs)
        time, data = scope.read()
        # plot the recorded data
        for p, d in enumerate(data):
            lb = scope.labels[p] if p < len(scope.labels) else f"port {p}"
            plt.plot(time, d, label=lb)
        plt.legend()
    plt.xlabel("Time")
    plt.show()
