# New launcher, much easier to manipulate than a GUI as it's purely text editable

import os
import sys
import subprocess
import argparse
import requests
import time
import math
import json
import shutil
import copy

pwd = os.getcwd()

parser = argparse.ArgumentParser(description="New StableTuner Launcher")
parser.add_argument("--config", help="Where the config is, if the file doesn't exist, will generate a template.")
parser.add_argument("--webhook", default="", help="The Discord webhook that you want to emit finished training runs to.")
parser.add_argument("--show_unchanged_settings", action="store_true", help="Shows which config lines are redundant.")
parser.add_argument("--hide_ignored_settings", action="store_true", help="Cleans up output for messy or broken configs.")
parser.add_argument("--no_exec", action="store_true", help="Disables execution of subprocesses and file copy ops")
parser.add_argument("--resume_ccosine", type=int, default=-1, help="Resumes constant cosine from supplied epoch value.")
parser.add_argument("--convert", action="store_true", help="Converts diffusers to epochs")

args = parser.parse_args()

st_args = {} # Defaults to compare against
st_settings = {} # From config file
st_comments = {} # Comments for example writing

# Common Types: bool int str float
def register_arg(name, default, datatype, comment):
	global st_args
	global st_comments

	st_args[name] = default
	st_args[name+"_type"] = datatype
	if default == "" or default == -1:
		st_comments[name] = comment + f"\n# Type: {datatype}\n# Default: none"
	else:
		st_comments[name] = comment + f"\n# Type: {datatype}\n# Default: {default}"


# The big list of defaults;

# Project Settings:
st_comments["header0"] = "Project Settings:"
register_arg("project_name", "model", "str", "The model name for your training run.")
register_arg("project_append", "", "str", "What gets added after the project name and epoch.")

# Training Settings:
st_comments["header1"] = "Training Settings:"
register_arg("resolution", 512, "int", "The resolution for all images, and all images will be resied to this resolution.")
register_arg("train_batch_size", 4, "int", "Batch size (per connected HF Accelerate device) for training with.")
register_arg("num_train_epochs", 1, "int", "The number of epochs to train.")
register_arg("shuffle_per_epoch", True, "bool", "Shuffles the order of the batches every epoch.")
register_arg("use_bucketing", False, "bool", "Whether to use square crops or using the whole image with bucket based resolution.")
register_arg("seed", 42, "int", "A seed for reproducible training.")
register_arg("learning_rate", 5e-6, "float", "Initial learning rate (after the a supplied warming period) to use.")
register_arg("lr_scheduler", "constant", "str", "The scheduler type to use. Choose between:\n# linear\n# cosine\n# cosine_with_restarts\n# polynomial\n# constant\n# constant_with_warmup\n# constant_cosine - handled by the launcher.")
register_arg("lr_warmup_steps", 500, "int", "The number of steps for the warmup in the LR scheduler.")
register_arg("token_limit", 75, "int", "Token limit, token lengths longer than the next multiple of 75 will be truncated.")
register_arg("epoch_seed", False, "bool", "Increments the seed on every new epoch.")
register_arg("min_snr_gamma", "", "float", "Gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper.")
register_arg("with_pertubation_noise", False, "bool", "Enhancement that increases how fast convergence occurs.")
register_arg("pertubation_noise_weight", 0.1, "float", "The weight of pertubation noise applied during training.")
register_arg("zero_terminal_snr", False, "bool", "Enables Zero Terminal SNR, see https://arxiv.org/pdf/2305.08891.pdf - requires force_v_pred for non SD2.1 models.")
register_arg("force_v_pred", False, "bool", "Force enables V Prediction for models that don't officially support it - ie SD1.x.")
register_arg("scale_v_pred_loss", False, "bool", "By scaling the loss according to the time step, the weights of global noise prediction and local noise prediction become the same, and the improvement of details may be expected.")
register_arg("conditional_dropout", -1, "float", "A percentage of batches to drop out. 0 uses none, 1 uses all of them. Use none to disable. Text encoder and unet are not trained with captions when dropped out.")

# Model Settings
st_comments["header2"] = "Model Settings:"
register_arg("model_variant", "base", "str", "Which type of model we're training, base for basic txt2img, img2img for training the img2img portion and uses masking, and depth2img for training SD2.x models.")
register_arg("train_text_encoder", False, "bool", "Whether to train the text encoder or not.")
register_arg("stop_text_encoder_training", 999999999999999, "int", "Which epoch should the text encoder stop training at?")
register_arg("clip_penultimate", False, "bool", "Clip Skip 2 training for tag based models based on boorus.")
register_arg("pretrained_model_name_or_path", "", "str", "Path to pretrained model or model identifier from huggingface.co/models.\n# Defaults (Pick One):\n# CompVis/stable-diffusion-v1-4\n# runwayml/stable-diffusion-v1-5\n# runwayml/stable-diffusion-inpainting\n# stabilityai/stable-diffusion-2-base\n# stabilityai/stable-diffusion-2\n# stabilityai/stable-diffusion-2-inpainting\n# stabilityai/stable-diffusion-2-depth\n# stabilityai/stable-diffusion-2-1-base\n# stabilityai/stable-diffusion-2-1")
register_arg("pretrained_vae_name_or_path", "", "str", "Path to pretrained vae or vae identifier from huggingface.co/models.")
register_arg("tokenizer_name", "", "str", "Pretrained tokenizer name or path if not the same as model_name.")
register_arg("use_ema", False, "bool", "Whether or not to use Exponential Moving Average weighting over regular weights. Consumes more memory, slows training performance slightly.")
register_arg("disable_text_encoder_after", 999, "int", "Disables text encoder training after a specified epoch for constant_cosine.")

# Dataset Settings
st_comments["header3"] = "Dataset Settings:"
register_arg("concepts_list", "", "str", "Path to json containing one or more concepts. This option will overwrite others like instance_prompt, etc.")
register_arg("aspect_mode", "dynamic", "str", "dynamic will choose to truncate or add duplicate images to fill out the batches evenly as possible. add will only duplicate existing images. truncate will remove images to fit the batch size.")
register_arg("aspect_mode_action_preference", "add", "str", "Overrides aspect_mode, but takes the same values for the same functions.")
register_arg("dataset_repeats", 1, "int", "How many times all the images in the dataset will be repeated.")
register_arg("use_text_files_as_captions", False, "bool", "Whether to use text files as captions within your dataset. Example: image.png image.txt in the same folder together.")
register_arg("use_image_names_as_captions", False, "bool", "Use image names themselves as captions.")
register_arg("auto_balance_concept_datasets", False, "bool", "This will equalise all datasets to the length of the smallest one. Destructive otherwise.")

# Optimisations
st_comments["header4"] = "Optimisations:"
register_arg("mixed_precision", "no", "str", "Whether to use mixed precision. Supported values: no, fp16, bf16, tf32. bf16 requires PyTorch >= 1.10 and an NVIDIA Ampere GPU.")
register_arg("attention", "xformers", "str", "The type of attention to use, xformers is for NVIDIA cards, whereas flash_attention is for any other type of card.")
register_arg("disable_cudnn_benchmark", True, "bool", "Disables the benchmark built into CuDNN. Always leave this enabled.")
register_arg("gradient_accumulation_steps", 1, "int", "Number of updates steps to accumulate before performing a backwards/update pass.")
register_arg("gradient_checkpointing", False, "bool", "Whether or not to use gradient checkpointing to save memory at the expense of a slower backwards/update pass.")
register_arg("scale_lr", False, "bool", "Scale the learning rate by the number of active GPUs, gradient accumulation steps and batch size.")
register_arg("use_8bit_adam", False, "bool", "Whether or not to use 8-bit Adam from bitsandbytes. Required on NVIDIA cards with 24GBs of VRAM or less for finetuning.")
register_arg("adam_beta1", 0.9, "float", "The beta1 parameter for the Adam optimiser.")
register_arg("adam_beta2", 0.999, "float", "The beta2 parameter for the Adam optimiser.")
register_arg("adam_weight_decay", 1e-2, "float", "Weight decay for the Adam optimiser.")
register_arg("adam_epsilon", 1e-8, "float", "Epsilon value for the Adam optimiser.")
register_arg("max_grad_norm", 1.0, "float", "Maximum gradient norm.")
register_arg("use_deepspeed_adam", False, "bool", "Use experimental DeepSpeed Adam 8bit.")
register_arg("use_torch_compile", False, "bool", "Whether to compile certain functions with Torch 2.")

# Misc Settings
st_comments["header5"] = "Misc Settings:"
register_arg("save_every_n_epoch", 1, "int", "When to save after each epoch completes, 1 will save all epochs, 2 will save every other epoch, and so forth.")
register_arg("save_every_quarter", False, "bool", "Whether the model should be saved every 25 percent of steps in an epoch.")
register_arg("output_dir", "", "str", "The output directory where the model checkpoints, logs and latent caches will be saved. This can be anywhere on your computer.")
register_arg("max_train_steps", -1, "int", "Total number of training steps to perform, if provided, overrides num_train_epochs.")
register_arg("regenerate_latent_cache", False, "bool", "Regenerates the latent cache, even if it already exists. Very dangerous.")
register_arg("use_latents_only", False, "bool", "Runs from an existing latent cache, last_run.json is checked for compliance.")
register_arg("logging_dir", "logs", "str", "The TensorBoard and CSV log directory. Will default to output_dir/logs/logs.")
register_arg("log_interval", 10, "int", "Log every N steps.")
register_arg("overwrite_csv_logs", False, "bool", "Overwrites the CSV containing loss, LR, current step, current epoch and a timestamp when starting a new training session.")
register_arg("local_rank", 1, "int", "For distributed training: local_rank.")
register_arg("detect_full_drive", True, "bool", "Detects when the storage drive is full and deletes older checkpoints.")

# Dreambooth Settings
st_comments["header6"] = "Dreambooth Settings:"
register_arg("with_prior_preservation", False, "bool", "Enable this to add prior preservation loss / Dreambooth.")
register_arg("prior_loss_weight", 1.0, "float", "The weight of prior preservation loss.")

# Legacy Settings
st_comments["header7"] = "Legacy Settings:"
register_arg("num_class_images", 100, "int", "LEGACY: Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class prompt.")
register_arg("center_crop", False, "bool", "LEGACY: Whether to center crop images before resizing to resolution.")
register_arg("instance_data_dir", "", "str", "LEGACY: A folder containing the training data of instance images.")
register_arg("instance_prompt", "", "str", "LEGACY: The prompt with identifier specifying the instance.")
register_arg("class_data_dir", "", "str", "LEGACY: A folder containing the training data of class images.")
register_arg("class_prompt", "", "str", "LEGACY: The prompt to specify images in the same class as provided instance images.")
register_arg("add_mask_prompt", "", "str", "LEGACY: The prompt to specify images for inpainting - not properly supported.")
register_arg("sample_batch_size", 4, "int", "LEGACY: Batch size (per connected HF Accelerate device) for sampling images.")
register_arg("add_class_images_to_dataset", False, "bool", "LEGACY: This adds your regularisation images (class images) to your training dataset.")

# User config:
# Basic tag pruning lists
if not os.path.isfile(args.config):
	with open(args.config, "w", encoding="utf-8") as file:
		data = "### Config Information:\n"
		data += "### Lines starting with a # are not used by the config loader.\n### Lines that match the internal default settings will be ignored, so if batch_size were to be 4 in your config,\n### it would be ignored, whereas a value of 20 would be used instead.\n"
		data += "### Items with a default of none can just be written as 'min_snr_gamma,'\n"
		data += "### or as 'min_snr_gamma,none'\n\n"
		for key in st_comments.keys():
			data += f"## {st_comments[key]}\n"
			if key not in st_args:
				continue

			if isinstance(st_args[key], bool):
				if st_args[key]:
					data += f"{key},true\n\n"
				else:
					data += f"{key},false\n\n"
			else:
				data += f"{key},{st_args[key]}\n\n"
		file.write(data)
	print(f"Example config file written to {args.config}, stopping.")
	exit()

# Parse config
with open(args.config, "r", encoding="utf-8") as config:
	settings = config.readlines()
	for config in settings:
		# Skip commented lines
		if config.startswith("#") or config.strip() == "":
			continue
		conf = config.lower().strip().split(",", 1)
		# Only do matching arguments
		if conf[0] in st_args:
			conf_value = conf[1]
			if st_args[conf[0] + "_type"] == "int":
				try:
					conf_value = int(conf[1])
				except:
					print(f"{conf[0]} has an invalid type, should be integer, but isn't. Setting ignored.")
			elif st_args[conf[0] + "_type"] == "float":
				try:
					conf_value = float(conf[1])
				except:
					print(f"{conf[0]} has an invalid type, should be float, but isn't. Setting ignored.")
			else: # bool, str
				if conf[1] == "" or conf[1] == "none":
					conf_value = st_args[conf[0]]
				elif conf[1] == "true":
					conf_value = True
				elif conf[1] == "false":
					conf_value = False
				else:
					conf_value = conf[1]

			if st_args[conf[0]] != conf_value:
				st_settings[conf[0]] = conf_value
			elif args.show_unchanged_settings:
				print(f"{conf[0]} has no change in actual value; ignoring.")
		elif not args.hide_ignored_settings:
			print(f"{conf[0]} matches no existing setting; ignoring.")

if "output_dir" not in st_settings:
	print("Stopping, no valid output directory.")
	exit()
else:
	# Strip any directory names ending with a slash
	if st_settings["output_dir"][len(st_settings["output_dir"])-1] == "/" or st_settings["output_dir"][len(st_settings["output_dir"])-1] == "\\":
		st_settings["output_dir"] = st_settings["output_dir"][:-1]

if "project_name" not in st_settings:
	st_settings["project_name"] = st_args["project_name"]

if "project_append" not in st_settings:
	st_settings["project_append"] = st_args["project_append"]

# Generate the launch command for StableTuner
launcher_args = ["accelerate", "launch", '--mixed_precision=fp16', "scripts/trainer.py"]

skip_these_settings = {"project_name": True, "project_append": True, "disable_text_encoder_after": True}
def parse_settings(settings):
	global launcher_args
	launcher_args = ["accelerate", "launch", '--mixed_precision=fp16', "scripts/trainer.py"]
	for setting in settings.keys():
		# Skip args that shouldn't be used to launch ST, namely project settings
		if setting in skip_these_settings:
			continue

		if st_args[f"{setting}_type"] == "bool":
			launcher_args.append(f"--{setting}")
		#elif st_args[f"{setting}_type"] == "float" or st_args[f"{setting}_type"] == "int":
			#launcher_args.append(f'--{setting}={st_settings[setting]}')#
		else:
			launcher_args.append(f'--{setting}={st_settings[setting]}')
		

# constant_cosine is a special case
are_we_constant_cosine = False
start_lr = 0
if "lr_scheduler" in st_settings:
	if st_settings["lr_scheduler"] == "constant_cosine":
		are_we_constant_cosine = True
		st_settings["lr_scheduler"] = "constant"
		if "learning_rate" in st_settings:
			start_lr = copy.deepcopy(st_settings["learning_rate"])
		else:
			start_lr = copy.deepcopy(st_args["learning_rate"])

def cosine_curve(epoch, total_epochs):
	global start_lr
	pi = math.pi
	total_steps = 123456789

	perc_done = epoch / total_epochs
	if epoch == 0:
		perc_done = 0
	elif epoch == total_epochs:
		perc_done = (total_steps-1)/total_steps

	# This is a very dirty Python hack but I'll allow it on the grounds of we don't really need more than two decimal places
	new_lr = float("{:.3g}".format(start_lr*(0.5*(1+math.cos(pi*0.5*2*perc_done)))))
	return new_lr

if are_we_constant_cosine:
	max_epochs = 1
	if "num_train_epochs" not in st_settings:
		print("Not enough epochs to use constant cosine with. Must be more than one epoch, stopping.")
		exit()
	elif st_settings["num_train_epochs"] < 2:
		print("Not enough epochs to use constant cosine with. Must be more than one epoch, stopping.")
		exit()
	else:
		max_epochs = copy.deepcopy(st_settings["num_train_epochs"])

	if "seed" not in st_settings:
		st_settings["seed"] = st_args["seed"]

	# Important file things for automated uploads
	input_diffusers = f'{st_settings["output_dir"]}/epoch_1'
	output_filename = ""
	output_checkpoint = f'{st_settings["output_dir"]}'
	output_path = ""
	# We only want to train one epoch at a time
	st_settings["num_train_epochs"] = 1
	for e in range(max_epochs):
		# Handle the cosine decay
		st_settings["learning_rate"] = cosine_curve(e, max_epochs)
		parse_settings(st_settings)
		output_filename = f'{st_settings["project_name"]}_e{e+1}_{st_settings["project_append"]}.safetensors'
		output_checkpoint = f'{st_settings["output_dir"]}/{output_filename}'
		output_path = f'{st_settings["output_dir"]}/{st_settings["project_name"]}_e{e+1}_{st_settings["project_append"]}'

		if e >= args.resume_ccosine-1:
			# Debug printing
			if args.no_exec:
				print(f"Using pretrained_model_name_or_path: {st_settings['pretrained_model_name_or_path']}")
			
			if not args.no_exec:
				print(f"Now training Epoch {e+1}.")
				# Train the epoch
				subprocess.run(launcher_args)
				if args.webhook != "":
					print(f"\n\nTraining Epoch {e+1} completed, converting to safetensors now.")
					time.sleep(3)
					# Convert the epoch
					subprocess.run(["python", "scripts/convert_diffusers_to_sd_cli.py", input_diffusers, output_checkpoint])
					# Move the diffusers folder to safety
					shutil.move(input_diffusers, output_path)
			else:
				# More debug information
				print(f"Epoch: {e+1}, LR: {st_settings['learning_rate']}, Seed: {st_settings['seed']}, CKPT: {output_filename}")

		# Process model configuration
		if "epoch_seed" in st_settings:
			if st_settings["epoch_seed"]:
				st_settings["seed"] += 1
		if "use_latents_only" not in st_settings:
			st_settings["use_latents_only"] = True
			print("Running from latent cache only.")
		elif not st_settings["use_latents_only"]:
			st_settings["use_latents_only"] = True
			print("Running from latent cache only.")
		if e+1 == st_settings["disable_text_encoder_after"] and st_settings["train_text_encoder"]:
			st_settings["train_text_encoder"] = False
			print("Disabling text encoder training.")
		st_settings["pretrained_model_name_or_path"] = output_path

		if e >= args.resume_ccosine-1:
			if args.no_exec:
				print(f"Diffusers: {input_diffusers}, Rename: {output_path}\n")

			if args.webhook != "" and not args.no_exec:
				print("Now preparing upload to PixelDrain.")
				file = open(output_checkpoint, "rb")
				pixeldrain_api = "https://pixeldrain.com/api/file"
				pixeldrain_response = requests.post(pixeldrain_api, files = {"file": file, "name": output_filename, "anonymous": True})
				pixeldrain_json = pixeldrain_response.json()
				if pixeldrain_json["success"]:
					data = {"content": f"# New Checkpoint! :tada:\n\n{output_filename}:\nhttps://pixeldrain.com/u/{pixeldrain_json['id']}", "username": "Fluffusion Trainer"}
					webhook = requests.post(args.webhook, json=data)
					print(f"Uploaded to PixelDrain as: https://pixeldrain.com/u/{pixeldrain_json['id']}")
				else:
					data = {"content": f"PixelDrain is down or something happened during upload. :(\nReason: {pixeldrain_json['message']}\nType: {pixeldrain_json['value']}", "username": "Fluffusion Trainer"}
					webhook = requests.post(args.webhook, json=data)

else:
	parse_settings(st_settings)

	if not args.no_exec:
		subprocess.run(launcher_args)

	max_epochs = 1
	if "num_train_epochs" in st_settings:
		max_epochs = copy.deepcopy(st_settings["num_train_epochs"])

	# Important file things for automated uploads
	input_diffusers = f'{st_settings["output_dir"]}/epoch_{max_epochs}'
	if "save_every_n_epoch" in st_settings:
		if st_settings["save_every_n_epoch"] != 1:
			input_diffusers = f'{st_settings["output_dir"]}/epoch_{max_epochs+1}'
	output_filename = f'{st_settings["project_name"]}_e{max_epochs}_{st_settings["project_append"]}.safetensors'
	dest_file = f'{st_settings["output_dir"]}/{output_filename}'
	if not args.no_exec or args.convert:
		subprocess.run(["python", "scripts/convert_diffusers_to_sd_cli.py", input_diffusers, dest_file])
		# Move the diffusers folder to safety
		# shutil.move(input_diffusers, f'{st_settings["output_dir"]}/{st_settings["project_name"]}_e{max_epochs}_{st_settings["project_append"]}')

		if args.webhook != "":
			file = open(dest_file, "rb")
			pixeldrain_api = "https://pixeldrain.com/api/file"
			pixeldrain_response = requests.post(pixeldrain_api, files = {"file": file, "name": output_filename, "anonymous": True})
			pixeldrain_json = pixeldrain_response.json()
			if pixeldrain_json["success"]:
				data = {"content": f"# New Checkpoint! :tada:\n\n{output_filename}:\nhttps://pixeldrain.com/u/{pixeldrain_json['id']}", "username": "Fluffusion Trainer"}
				webhook = requests.post(args.webhook, json=data)
			else:
				data = {"content": f"PixelDrain is down or something happened during upload. :(", "username": "Fluffusion Trainer"}
				webhook = requests.post(args.webhook, json=data)