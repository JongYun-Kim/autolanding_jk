from gym.envs.registration import register

register(
    id='vision-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VisionAviary',
)

register(
    id='takeoff-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:TakeoffAviary',
)

register(
    id='landing-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:LandingAviary',
)

register(
    id='gimbal-landing-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl.LandingAviary:LandingGimbalAviary',
)
register(
    id='gimbal-curriculum-landing-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl.LandingAviary:LandingGimbalCurriculumAviary',
)