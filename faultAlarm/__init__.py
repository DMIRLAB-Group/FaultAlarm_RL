from gym.envs.registration import register

register(
    id='FaultAlarm-v0',
    entry_point='faultAlarm.envs:FaultAlarmEnv',
)