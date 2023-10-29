class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing
    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(
            self,
            concept_list,
            seed=555,
            debug_level=0,
            resolution=512,
            batch_size=1,
            flip_p=0.0,
            use_image_names_as_captions=True,
            add_class_images_to_dataset=False,
            balance_datasets=False,
            with_prior_loss=False,
            use_text_files_as_captions=False,
            aspect_mode='dynamic',
            action_preference='add',
            model_variant='base',
            extra_module=None,
            mask_prompts=None,
    ):
        self.resolution = resolution
        self.debug_level = debug_level
        self.flip_p = flip_p
        self.use_image_names_as_captions = use_image_names_as_captions
        self.balance_datasets = balance_datasets
        self.with_prior_loss = with_prior_loss
        self.add_class_images_to_dataset = add_class_images_to_dataset
        self.use_text_files_as_captions = use_text_files_as_captions
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        self.seed = seed
        self.model_variant = model_variant
        self.extra_module = extra_module
        self.prepared_train_data = []
        prepared_train_data = []
        
        self.aspects = get_aspect_buckets(resolution)
        #print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        #process sub directories flag
            
        print(f"{bcolors.WARNING} Preloading images...{bcolors.ENDC}")   

        if balance_datasets:
            print(f"{bcolors.WARNING} Balancing datasets...{bcolors.ENDC}") 
            #get the concept with the least number of images in instance_data_dir
            min_concept = min(concept_list, key=lambda x: len(os.listdir(x['instance_data_dir'])))
            #get the number of images in the concept with the least number of images
            min_concept_num_images = len(os.listdir(min_concept['instance_data_dir']))
            print(" Min concept: ",min_concept['instance_data_dir']," with ",min_concept_num_images," images")
            
            balance_cocnept_list = []
            for concept in concept_list:
                #if concept has a key do not balance it
                if 'do_not_balance' in concept:
                    if concept['do_not_balance'] == True:
                        balance_cocnept_list.append(-1)
                    else:
                        balance_cocnept_list.append(min_concept_num_images)
                else:
                        balance_cocnept_list.append(min_concept_num_images)
        for concept in concept_list:
            if 'use_sub_dirs' in concept:
                if concept['use_sub_dirs'] == True:
                    use_sub_dirs = True
                else:
                    use_sub_dirs = False
            else:
                use_sub_dirs = False
            self.image_paths = []
            #self.class_image_paths = []
            min_concept_num_images = None
            if balance_datasets:
                min_concept_num_images = balance_cocnept_list[concept_list.index(concept)]
            data_root = concept['instance_data_dir']
            data_root_class = concept['class_data_dir']
            concept_prompt = concept['instance_prompt']
            concept_class_prompt = concept['class_prompt']
            if 'flip_p' in concept.keys():
                flip_p = concept['flip_p']
                if flip_p == '':
                    flip_p = 0.0
                else:
                    flip_p = float(flip_p)
            self.__recurse_data_root(self=self, recurse_root=data_root,use_sub_dirs=use_sub_dirs)
            random.Random(self.seed).shuffle(self.image_paths)
            if self.model_variant == 'depth2img':
                print(f"{bcolors.WARNING} ** Loading Depth2Img Pipeline To Process Dataset{bcolors.ENDC}")
                self.vae_scale_factor = self.extra_module.depth_images(self.image_paths)
            prepared_train_data.extend(self.__prescan_images(debug_level, self.image_paths, flip_p,use_image_names_as_captions,concept_prompt,use_text_files_as_captions=self.use_text_files_as_captions)[0:min_concept_num_images]) # ImageTrainItem[]
            if add_class_images_to_dataset:
                self.image_paths = []
                self.__recurse_data_root(self=self, recurse_root=data_root_class,use_sub_dirs=use_sub_dirs)
                random.Random(self.seed).shuffle(self.image_paths)
                use_image_names_as_captions = False
                prepared_train_data.extend(self.__prescan_images(debug_level, self.image_paths, flip_p,use_image_names_as_captions,concept_class_prompt,use_text_files_as_captions=self.use_text_files_as_captions)) # ImageTrainItem[]
            
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        if self.with_prior_loss and add_class_images_to_dataset == False:
            self.class_image_caption_pairs = []
            for concept in concept_list:
                self.class_images_path = []
                data_root_class = concept['class_data_dir']
                concept_class_prompt = concept['class_prompt']
                self.__recurse_data_root(self=self, recurse_root=data_root_class,use_sub_dirs=use_sub_dirs,class_images=True)
                random.Random(seed).shuffle(self.image_paths)
                if self.model_variant == 'depth2img':
                    print(f"{bcolors.WARNING} ** Depth2Img To Process Class Dataset{bcolors.ENDC}")
                    self.vae_scale_factor = self.extra_module.depth_images(self.image_paths)
                use_image_names_as_captions = False
                self.class_image_caption_pairs.extend(self.__prescan_images(debug_level, self.class_images_path, flip_p,use_image_names_as_captions,concept_class_prompt,use_text_files_as_captions=self.use_text_files_as_captions))
            self.class_image_caption_pairs = self.__bucketize_images(self.class_image_caption_pairs, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        if self.model_variant == "inpainting" and mask_prompts is not None:
            print(f"{bcolors.WARNING} Checking and generating missing masks...{bcolors.ENDC}")
            clip_seg = ClipSeg()
            clip_seg.mask_images(self.image_paths, mask_prompts)
            del clip_seg
        if debug_level > 0: print(f" * DLMA Example: {self.image_caption_pairs[0]} images")
        #print the length of image_caption_pairs
        print(f"{bcolors.WARNING} Number of image-caption pairs: {len(self.image_caption_pairs)}{bcolors.ENDC}") 
        if len(self.image_caption_pairs) == 0:
            raise Exception("All the buckets are empty. Please check your data or reduce the batch size.")
    def get_all_images(self):
        if self.with_prior_loss == False:
            return self.image_caption_pairs
        else:
            return self.image_caption_pairs, self.class_image_caption_pairs
    def __prescan_images(self,debug_level: int, image_paths: list, flip_p=0.0,use_image_names_as_captions=True,concept=None,use_text_files_as_captions=False):
        """
        Create ImageTrainItem objects with metadata for hydration later 
        """
        decorated_image_train_items = []
        
        for pathname in tqdm(image_paths, desc="Pre-scanning Images"):
            identifier = concept 
            if use_image_names_as_captions:
                caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]
                identifier = caption_from_filename
            if use_text_files_as_captions:
                txt_file_path = os.path.splitext(pathname)[0] + ".txt"

                if os.path.exists(txt_file_path):
                    try:
                        with open(txt_file_path, 'r',encoding='utf-8',errors='ignore') as f:
                            identifier = f.readline().rstrip()
                            f.close()
                            if len(identifier) < 1:
                                raise ValueError(f" *** Could not find valid text in: {txt_file_path}")
                            
                    except Exception as e:
                        print(f"{bcolors.FAIL} *** Error reading {txt_file_path} to get caption, falling back to filename{bcolors.ENDC}") 
                        print(e)
                        identifier = caption_from_filename
                        pass
            #print("identifier: ",identifier)
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            image_train_item = ImageTrainItem(image=None, extra=None, caption=identifier, target_wh=target_wh, pathname=pathname, flip_p=flip_p,model_variant=self.model_variant)

            decorated_image_train_items.append(image_train_item)
        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0,aspect_mode='dynamic',action_preference='add'):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """

        # TODO: this is not terribly efficient but at least linear time
        buckets = {}
        for image_caption_pair in tqdm(prepared_train_data, desc="Preparing buckets"):
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair)
        print(f" ** Number of buckets: {len(buckets)}")
        for bucket in buckets:
            bucket_len = len(buckets[bucket])
            #real_len = len(buckets[bucket])+1
            #print(real_len)
            truncate_amount = bucket_len % batch_size
            add_amount = batch_size - bucket_len % batch_size
            action = None
            bratio = ""
            bmode = ""
            if bucket[0] <= bucket[1]:
                bratio = bucket[1] / bucket[0]
                if bratio == 1:
                    bmode = f"(1:1)"
                else:
                    bmode = f"(1:{bratio:.2f})"
            else:
                bratio = bucket[0] / bucket[1]
                bmode = f"({bratio:.2f}:1)"
            #print(f" ** Bucket {bucket} has {bucket_len} images")
            if aspect_mode == 'dynamic':
                if batch_size == bucket_len:
                    action = None
                elif add_amount < truncate_amount and add_amount != 0 and add_amount != batch_size or truncate_amount == 0:
                    action = 'add'
                    #print(f'should add {add_amount}')
                elif truncate_amount < add_amount and truncate_amount != 0 and truncate_amount != batch_size and batch_size < bucket_len:
                    #print(f'should truncate {truncate_amount}')
                    action = 'truncate'
                    #truncate the bucket
                elif truncate_amount == add_amount:
                    if action_preference == 'add':
                        action = 'add'
                    elif action_preference == 'truncate':
                        action = 'truncate'
                elif batch_size > bucket_len:
                    action = 'add'

            elif aspect_mode == 'add':
                action = 'add'
            elif aspect_mode == 'truncate':
                action = 'truncate'
            if action == None:
                action = None
                #print('no need to add or truncate')
            if action == None:
                #print('test')
                current_bucket_size = bucket_len
                print(f"  ** Bucket {bucket} {bmode} found {bucket_len}, nice!")
            elif action == 'add':
                #copy the bucket
                shuffleBucket = random.sample(buckets[bucket], bucket_len)
                #add the images to the bucket
                current_bucket_size = bucket_len
                truncate_count = (bucket_len) % batch_size
                #how many images to add to the bucket to fill the batch
                addAmount = batch_size - truncate_count
                if addAmount != batch_size:
                    added=0
                    while added != addAmount:
                        randomIndex = random.randint(0,len(shuffleBucket)-1)
                        #print(str(randomIndex))
                        buckets[bucket].append(shuffleBucket[randomIndex])
                        added+=1
                    print(f"  ** Bucket {bucket} {bmode} found {bucket_len}  images, will {bcolors.OKCYAN}duplicate {added} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
                else:
                    print(f"  ** Bucket {bucket} {bmode} found {bucket_len}, {bcolors.OKGREEN}nice!{bcolors.ENDC}")
            elif action == 'truncate':
                truncate_count = (bucket_len) % batch_size
                current_bucket_size = bucket_len
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                print(f"  ** Bucket {bucket} found {bucket_len} ({bmode}) images, will {bcolors.FAIL}drop {truncate_count} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
            

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        progress_bar = tqdm(os.listdir(recurse_root), desc=f"{bcolors.WARNING} ** Processing {recurse_root}{bcolors.ENDC}")
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if '-depth' in f or '-masklabel' in f:
                    progress_bar.update(1)
                    continue
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    #try to open the file to make sure it's a valid image
                    try:
                        img = Image.open(current)
                    except:
                        print(f" ** Skipping {current} because it failed to open, please check the file")
                        progress_bar.update(1)
                        continue
                    del img
                    if class_images == False:
                        self.image_paths.append(current)
                    else:
                        self.class_images_path.append(current)
            progress_bar.update(1)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(recurse_root):
                current = os.path.join(recurse_root, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                self.__recurse_data_root(self=self, recurse_root=dir)