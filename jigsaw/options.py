### Taken from Raven ###

import click

### HELPERS/CALLBACKS ###
def no_user_callback(ctx, param: click.core.Option, value: bool):
    """Callback used by the no-user option. Evaluates all loaded parameters
    and if any are still None, throws an error.

    Args:
        param (click.core.Option): option callback has been called by
        value (bool): value of no-user option, default false
    """
    # set in context
    if ctx.obj is None:
        ctx.obj = {}
    ctx.obj['NO_USER'] = value
    # if we are in no_user mode, check all required arguments are there
    if value:
        for arg, value in ctx.params.items():
            if value is None:
                # ctx.exit('You must supply the --%s argument when using --no-user!'%arg)
                raise click.exceptions.BadParameter('You must provide this argument when using --no-user!',
                                                    ctx=ctx, param=arg, param_hint=str(arg))


### OPTIONS ###
# NOTE: Any option intended to be used alongside the no_user_opt on a command
# must have their is_eager flag set to TRUE for no_user_opt to work properly
"""Flag to determine if the command should run in user mode.
"""
no_user_opt = click.option(
    '--no-user', is_flag=True, callback=no_user_callback, expose_value=False,
    help='Disable Inquirer prompts and use flags instead.'
)

verbose_opt = click.option(
    '-v', '--verbose', is_flag=True,
    help='Run command in verbose mode.'
)

# Dataset Creation Options
training_type_opt = click.option(
    '-type', '-t', type=str, is_eager=True, help='training type being used on dataset'
)

bucket_opt = click.option(
    '-b', '--bucket', 'bucket', type=str, is_eager=True,
    help='Name of bucket data should be taken from. Ignored without --no-user.'
)

local_opt = click.option(
    '-l', '--local', 'local', type=str, is_eager=True,
    help='Do not download from S3 and instead download data from the provided filepath. Ignored without --no-user.'
)

folders_opt = click.option(
    '-f', '--folders', type=str, multiple=True, is_eager=True,
    help='Prefixes/Folders to be downloaded for dataset. Ignored with --no-user.'  
)

name_opt = click.option(
    '--name', '-n', type=str, is_eager=True,
    help='Name of dataset being created. Ignored without --no-user.'
)

kfolds_opt = click.option(
    '--kfolds', '-k', type=int, default=-1, is_eager=True,
    help='Number of folds in dataset. Ignored without --no-user.'
)

notes_opt = click.option(
    '--notes', '-m', type=str, is_eager=True,
    help='Add any notes or comments about this dataset. Ignored without --no-user.'
)

username_opt = click.option(
    '--username', '-u', type=str, is_eager=True,
    help='Enter your first and last name. Ignored without --no-user.'
)

upload_opt = click.option(
    '--upload', type=str, is_eager=True,
    help='Enter the bucket you would like to upload too. Ignored without --no-user.'
)

delete_local_opt = click.option(
    '--delete-local', '-d', 'delete_local', is_flag=True, is_eager=True,
    help='Enter your first and last name. Ignored without --no-user.'
)