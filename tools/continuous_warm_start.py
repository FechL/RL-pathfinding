"""Optional, explicit demonstration warm start; never used at runtime.

A continuous feedback reference teaches basic collision-aware driving.
SAC then optimizes the actor using environment rewards. These demonstrations
are not shortest-time labels and no claim of learning from scratch is made.
"""
import math
import numpy as np


def reference_action(observation):
    x,y=observation[:2]*.8
    error=math.atan2(y,x)
    v=.34*max(0.,math.cos(error))**2
    # Index 12 is the forward ray (16 rays start at -pi, after 4 other inputs).
    v=min(v,math.sqrt(max(0.,2*.8*(float(observation[12])*.8-.10))))
    v=min(v,2*math.hypot(x,y))
    return np.array([2*v/.38-1,np.clip(5*error,-4,4)/4],dtype=np.float32)


def warm_start(model,nav,seed=42,samples=30000,updates=6000,preview=None):
    import torch
    torch.manual_seed(seed)
    from continuous_env import ContinuousDriveEnv
    rng=np.random.default_rng(seed)
    env=ContinuousDriveEnv(nav,training=True)
    env.curriculum=1.
    def show(stage,force=False):
        if preview:
            preview.stage=stage
            preview.update(env,force=force)
            if preview.closed:raise SystemExit('Training dihentikan pada tahap persiapan.')
    observations=[];actions=[]
    # Include the actual mission and a mixture of complete driving snippets
    # and arbitrary continuous poses/headings/velocities throughout the map.
    obs,_=env.reset(seed=seed,options={'position':nav.start,'heading':0.})
    for i in range(samples):
        if i%100==0:
            obs,_=env.reset()
        action=reference_action(obs)
        observations.append(obs.copy());actions.append(action)
        next_obs,reward,terminated,truncated,info=env.step(action)
        show('Contoh gerak awal',force=terminated or truncated)
        info['TimeLimit.truncated']=truncated and not terminated
        model.replay_buffer.add(obs[None,:],next_obs[None,:],action[None,:],np.array([reward]),
                                np.array([terminated or truncated]),[info])
        obs=next_obs
        if terminated or truncated:obs,_=env.reset()
    x=torch.tensor(np.array(observations),device=model.device)
    y=torch.tensor(np.array(actions),device=model.device)
    parameters=list(model.actor.latent_pi.parameters())+list(model.actor.mu.parameters())
    optimizer=torch.optim.Adam(parameters,lr=8e-4)
    for index in range(updates):
        if index%100==0:show('Mempelajari contoh gerak',force=True)
        indices=torch.as_tensor(rng.integers(0,samples,512),device=model.device)
        prediction=torch.tanh(model.actor.mu(model.actor.latent_pi(x[indices])))
        loss=torch.square(prediction-y[indices]).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
        if model.log_ent_coef is not None:model.log_ent_coef.fill_(math.log(.002))
    model.actor.optimizer.state.clear()
    # Adapt both critics to the demonstrated continuous controls before asking
    # them to improve the actor. Actor weights remain fixed in this phase.
    for index in range(4000):
        if index%100==0:show('Menyiapkan penilaian Q',force=True)
        batch=model.replay_buffer.sample(128)
        with torch.no_grad():
            next_actions,log_prob=model.actor.action_log_prob(batch.next_observations)
            next_q=torch.cat(model.critic_target(batch.next_observations,next_actions),dim=1).min(dim=1,keepdim=True).values
            entropy=model.log_ent_coef.exp() if model.log_ent_coef is not None else model.ent_coef_tensor
            target=batch.rewards+(1-batch.dones)*model.gamma*(next_q-entropy*log_prob.reshape(-1,1))
        values=model.critic(batch.observations,batch.actions)
        critic_loss=sum(torch.nn.functional.mse_loss(value,target) for value in values)
        model.critic.optimizer.zero_grad();critic_loss.backward();model.critic.optimizer.step()
        with torch.no_grad():
            for source,destination in zip(model.critic.parameters(),model.critic_target.parameters()):
                destination.lerp_(source,model.tau)
    return dict(method='behaviour cloning of continuous feedback reference',samples=samples,
                updates=updates,seed=seed,final_mse=float(loss.detach().cpu()),
                critic_warmup_updates=4000,
                scope='basic driving demonstrations, not optimal path or fastest-turn labels')


def demonstration_anchor(model,weight=10.):
    """SAC + a reference-action regularizer to prevent catastrophic forgetting.

    Add the BC gradient just before the normal SAC actor optimizer step.
    The actor still receives its original Q/entropy gradients. No reference
    controller or action blending is used during evaluation or deployment.
    """
    import torch
    def before_step(optimizer,args,kwargs):
        if model.replay_buffer.size()<128:return
        with torch.enable_grad():
            obs=model.replay_buffer.sample(128).observations
            x,y=obs[:,0]*.8,obs[:,1]*.8
            error=torch.atan2(y,x)
            v=.34*torch.clamp(torch.cos(error),min=0).square()
            v=torch.minimum(v,torch.sqrt(torch.clamp(1.6*(obs[:,12]*.8-.10),min=0)))
            v=torch.minimum(v,2*torch.sqrt(x*x+y*y))
            desired=torch.stack([2*v/.38-1,torch.clamp(5*error,-4,4)/4],dim=1)
            predicted=torch.tanh(model.actor.mu(model.actor.latent_pi(obs)))
            parameters=list(model.actor.latent_pi.parameters())+list(model.actor.mu.parameters())
            gradients=torch.autograd.grad(weight*torch.square(predicted-desired).mean(),parameters)
            for parameter,gradient in zip(parameters,gradients):
                if parameter.grad is None:parameter.grad=gradient
                else:parameter.grad.add_(gradient)
    return model.actor.optimizer.register_step_pre_hook(before_step)
