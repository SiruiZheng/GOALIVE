import subprocess
import click
import concurrent.futures

def command_gen(task,params):
    command = 'python3 ~/sirui/goalive-icml/BHER/'+params['alg'][0]+'/experiment/train.py --num_cpu 1 --env_name '+task+' --logdir ~/sirui/goalive-icml/BHER/'+params['alg'][0]+'/experiment/'+params['alg'][1]+'/'+params['runname']+'/'+task+'/seed_'+params['seed']+' --n_batches ' +str(params['n_batches'])+' --n_cycles ' +str(params['n_cycles'])+' --n_epochs ' +str(params['n_epochs'])+' --n_test_rollouts ' +str(params['n_test_rollouts']) +' --gpu_use False --seed '+params['seed'] + ' --replay_strategy '+params['alg'][2]
    return command
def params_gen(old_params_group,runname,n_batches,n_cycles,n_epochs,n_test_rollouts,gpu_use):
    params_group = []
    for item in old_params_group:
        item['runname'] = runname
        item['n_batches'] = n_batches
        item['n_cycles'] = n_cycles
        item['n_epochs'] = n_epochs
        item['n_test_rollouts'] = n_test_rollouts
        item['gpu_use'] = gpu_use
        params_group.append(item)
    return params_group

def subprogram(command):
    # Replace this with the actual work your subprogram does
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result)
    # task=input[0]
    # param_group = input[1]
    
    # results = []
    # for params in param_group:
    #     command = command_gen(task,params)
    #    print(command)
    #     result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #    print(result)
    #    results.append(result)
    return result


@click.command()
@click.option('--runname', type=str, default='icml-rebuttal', help='the name of the experiment')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--n_batches', type=int, default=40, help='training batches per cycle')
@click.option('--n_cycles', type=int, default=50, help='cycle per epoch')
@click.option('--n_test_rollouts', type=int, default=20, help='number of test rollouts per epoch, each consists of rollout_batch_size rollouts')
@click.option('--gpu_use', type=bool, default=False, help='whether to use GPU')
def main(runname,n_epochs,n_batches,n_cycles,n_test_rollouts,gpu_use):
    tasklist = [ 'HandReach-v0', 'HandManipulateBlockRotateZ-v0',
            'HandManipulateEggRotate-v0',  'HandManipulatePenRotate-v0']
    old_params_group = [{'alg':['her','ddpg','none'],'seed':'0'},{'alg':['her','ddpg','none'],'seed':'1'},{'alg':['her','ddpg','none'],'seed':'2'},{'alg':['her','ddpg','none'],'seed':'3'},{'alg':['her','ddpg','none'],'seed':'4'},{'alg':['goalive_finite','goalive_finite','future'],'seed':'0'},{'alg':['goalive_finite','goalive_finite','future'],'seed':'1'},{'alg':['goalive_finite','goalive_finite','future'],'seed':'2'},{'alg':['goalive_finite','goalive_finite','future'],'seed':'3'},{'alg':['goalive_finite','goalive_finite','future'],'seed':'4'}]
    params_group = params_gen(old_params_group,runname,n_batches,n_cycles,n_epochs,n_test_rollouts,gpu_use)
    command_list = []
    for task in tasklist:
        for params in params_group:
            command_list.append(command_gen(task,params))
    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        # Map each subprogram to an argument
        results = executor.map(subprogram, command_list)

        # Process results if needed
        for result in results:
            print(f"Result: {result}")





if __name__ == '__main__':
    main()
