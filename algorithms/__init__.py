from algorithms.sac import SAC
from algorithms.curl import CURL
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.svea_ted import SVEA_TED
from algorithms.svea_cmid import SVEA_CMID

algorithm = {
	'sac': SAC,
	'curl': CURL,
	'drq': DrQ,
	'svea': SVEA,
	'svea_ted': SVEA_TED,
	'svea_cmid': SVEA_CMID,
}

def make_agent(obs_shape, action_shape, action_range, cfg):
	return algorithm[cfg.algorithm](obs_shape, action_shape, action_range, cfg)