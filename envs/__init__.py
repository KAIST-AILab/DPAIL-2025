from .walker import MultimodalWalker
from .ant import MultimodalAnt
from .antgoal import MultimodalAntGoal
from .halfcheetah import MultimodalHalfCheetah
from .maze2d import MultiGoalMaze2dLarge, MultiGoalMaze2dMedium

MultimodalEnvs = {
    'Walker2d-v3': MultimodalWalker,
    'Ant-v3': MultimodalAnt,
    'HalfCheetah-v3': MultimodalHalfCheetah,
    'AntGoal-v3': MultimodalAntGoal,
    'maze2d-medium-v1': MultiGoalMaze2dMedium,
    'maze2d-large-v1': MultiGoalMaze2dLarge,
}
