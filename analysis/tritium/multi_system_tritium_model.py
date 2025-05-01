#########################################################################################
##
##                                     ARC fuel cycle
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Source, Scope, Block
from pathsim.solvers import RKBS32, RKF21, RKCK54
from pathsim.blocks.mixed import Pulse, Step, SquareWave

from tritium_model import (
    k_top,
    k_wall,
    baby_model,
    gas_switch_deltatime,
)

# PARAMETERS ============================================================================

A_wall = baby_model.A_wall
A_top = baby_model.A_top
V_salt = baby_model.volume
TBR = baby_model.TBR
neutron_rate = baby_model.neutron_rate

TBR = TBR.to("particle/neutron").magnitude

A_top = A_top.to("m^2").magnitude
A_wall = A_wall.to("m^2").magnitude
V_salt = V_salt.to("m^3").magnitude

k_top = k_top.to("m/s").magnitude

k_top *= 1.1

k_wall = 4 * k_top
gas_switch_deltatime = gas_switch_deltatime.to("seconds").magnitude
neutron_rate = neutron_rate.to("neutrons/s").magnitude

total_duration = 50 * 24 * 3600
irradiation_time = 12 * 3600

f = 0.05
initial_piping_concentration = 5.5e9  # initial concentration in the piping
k_exchange = 10 * k_top


# BLOCKS ================================================================================


def salt_ode(x, u, t):
    # unpack states
    c_salt = x
    # unpack inputs
    neutron_rate_in, h2_in = u
    dc_dt = (
        1
        / V_salt
        * (TBR * neutron_rate_in - c_salt * (k_top * A_top + h2_in * k_wall * A_wall))
    )
    return dc_dt


def piping_ode(x, u, t):
    # unpack states
    c_piping = x
    # unpack inputs
    c_salt_in, h2_in = u
    dc_dt = f * c_salt_in * k_top * A_top - c_piping * h2_in * k_exchange
    return dc_dt


salt = ODE(func=salt_ode)

piping = ODE(func=piping_ode, initial_value=initial_piping_concentration)

neutron_rate_block = Pulse(
    amplitude=neutron_rate,
    T=total_duration,
    duty=irradiation_time / total_duration,
)
h2 = Step(amplitude=1, tau=gas_switch_deltatime)

Sco = Scope(labels=["salt", "piping", "neutron_rate", "h2"])

blocks = [neutron_rate_block, salt, piping, h2, Sco]


# CONNECTIONS ===========================================================================

connections = [
    Connection(salt, piping[0], Sco[0]),
    Connection(piping, Sco[1]),
    Connection(neutron_rate_block, salt[0], Sco[2]),
    Connection(h2, salt[1], piping[1], Sco[3]),
]


# SIMULATION ============================================================================

Sim = Simulation(
    blocks,
    connections,
    log=True,
    Solver=RKBS32,
    tolerance_lte_rel=1e-4,
    tolerance_lte_abs=1e-9,
)


if __name__ == "__main__":

    Sim.run(total_duration)

    # Sim.save("baby.mdl")

    # fig, ax = Sco.plot()

    # plt.show()

    import plotly.graph_objects as go

    # plot total plant inventory
    time, data = Sco.read()

    fig = go.Figure()
    for p, d in enumerate(data):
        lb = Sco.labels[p] if p < len(Sco.labels) else f"port {p}"
        fig.add_trace(go.Scatter(x=time, y=d, mode="lines", name=lb))

    fig.update_layout(
        legend_title="Components",
        font=dict(size=10),
        # yaxis_type="log",
        # xaxis_type="log",
        xaxis_title="Time [s]",
        yaxis_title="Block value [#]",
    )
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")
    fig.write_html("baby.html")
    # fig.show()
