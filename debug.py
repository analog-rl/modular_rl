from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym


mondir = "tmp.dir"
env = 'CartPole-v0'
video = False
agent = 'modular_rl.agentzoo.TrpoAgent'
seed = 7
use_hdf = False


env = make(env)
os.mkdir(mondir)
env.monitor.start(mondir, video_callable=None if video else VIDEO_NEVER)
agent_ctor = get_agent_cls(agent)
# update_argument_parser(parser, agent_ctor.options)
# if args.timestep_limit == 0:
#     args.timestep_limit = env_spec.timestep_limit
# cfg = args.__dict__
np.random.seed(seed)
agent = agent_ctor(env.observation_space, env.action_space, [])
# if use_hdf:
#     hdf, diagnostics = prepare_h5_file(args)
gym.logger.setLevel(logging.WARN)
timestep_limit
COUNTER = 0


def callback(stats):
    global COUNTER
    COUNTER += 1
    # Print stats
    print "*********** Iteration %i ****************" % COUNTER
    print tabulate(filter(lambda (k, v): np.asarray(v).size == 1, stats.items()))  # pylint: disable=W0110
    # Store to hdf5
    if args.use_hdf:
        for (stat, val) in stats.items():
            if np.asarray(val).ndim == 0:
                diagnostics[stat].append(val)
            else:
                assert val.ndim == 1
                diagnostics[stat].extend(val)
        if args.snapshot_every and ((COUNTER % args.snapshot_every == 0) or (COUNTER == args.n_iter)):
            hdf['/agent_snapshots/%0.4i' % COUNTER] = np.array(cPickle.dumps(agent, -1))
    # Plot
    if args.plot:
        animate_rollout(env, agent, min(500, timestep_limit))


run_policy_gradient_algorithm(env, agent, callback=callback, usercfg=cfg)

# if args.use_hdf:
#     hdf['env_id'] = env_spec.id
#     try:
#         hdf['env'] = np.array(cPickle.dumps(env, -1))
#     except Exception:
#         print "failed to pickle env"  # pylint: disable=W0703

env.monitor.close()