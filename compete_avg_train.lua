-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
local model_utils = require 'util.model_utils'
cudnn=require 'cudnn'
opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 200,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 50,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
   ngen = 2 ,                  -- number of generators to add to the game
   lambda_compete=0.5,          -- the weight of the competing objective
   ip='129.67.94.239',
   port=8000
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local ngen = opt.ngen
local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label =  1
local fake_label = 0

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then netG = defineG_unet_128(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers"  then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D )
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   --G = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   G=torch.load(  paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), 'binary' )
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
  print('define model netG...')
  G={}
  G.netG1 = defineG(input_nc, output_nc, ngf)
  G.relu=nn.ReLU()
  G.cosine=nn.CosineDistance()
  for i=2,ngen do
      G['netG'..i]=G.netG1:clone()
      G['netG'..i]:apply(weights_init)
  end
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

--print(G)
--print(netD)


local criterion = nn.BCECriterion()
--local criterion=cudnn.SpatialCrossEntropyCriterion()
local criterionAE = nn.AbsCriterion()
local compete_criterion=nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(ngen,opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(ngen,opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local score_D_cache=torch.Tensor(ngen,opt.batchSize )
local feature_cache=torch.Tensor(ngen,opt.batchSize,1*30*30)
local sum_score_D=torch.Tensor(opt.batchSize)

----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   score_D_cache=score_D_cache:cuda(); feature_cache=feature_cache:cuda(); sum_score_D=sum_score_D:cuda();
   if opt.cudnn==1 then
      --netG = util.cudnn(netG); 
      netD = util.cudnn(netD);
      for k,net in pairs(G) do
          G[k]=util.cudnn(net)
      end
   end
   netD:cuda();  criterion:cuda(); criterionAE:cuda(); compete_criterion:cuda()
   for k,net in pairs(G) do net:cuda() end
   print('done')
else
	print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
--local parametersG, gradParametersG = netG:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)



if opt.display then 
	disp = require 'display' 
	disp.configure({hostname=opt.ip,port=opt.port})
end


function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake
    for i=1,ngen do
        fake_B[i] = G['netG'..i]:forward(real_A)
        if opt.condition_GAN==1 then
            fake_AB[i] = torch.cat(real_A,fake_B[i],2)
        else
            fake_AB[i] = fake_B[i] -- unconditional GAN, only penalizes structure in B
        end
    end
    --local predict_real = netD:forward(real_AB)
    --local predict_fake = netD:forward(fake_AB)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    for i=1,ngen do 
        G['netG'..i]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    end
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor( output[{{} ,1  }]:size()  ):fill(real_label)
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    sum_score_D:zero()    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)
    local errD_fake=0 
    -- Fake
    for i=1,ngen do
        local output = netD:forward(fake_AB[i])
        label:fill(fake_label)
        errD_fake = errD_fake+ criterion:forward(output, label)
        score_D_cache[i]=output:reshape(opt.batchSize,1*30*30):sum(2)/(1*30*30)
        sum_score_D=sum_score_D+score_D_cache[i]
        feature_cache[i]=netD.modules[12].output:reshape(opt.batchSize,1*30*30)
        local df_do = criterion:backward(output, label)
        netD:backward(fake_AB[i], df_do)
    end
    errD = (errD_real + errD_fake)/(ngen+1)
    
    return errD, gradParametersD
end

local cosine_distance=function(feature_cache,k)
    local result=torch.Tensor(sum_score_D:size()):fill(0):cuda()
    for i=1,ngen do
        if i==k then
            goto continue
        else
            result=result+G.cosine:forward({ feature_cache[k],feature_cache[i]})
        end
        ::continue::
    end
    return result
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    for i=1,ngen do
        G['netG'..i]:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    end
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    errG=0 
    for i=1,ngen do
        if opt.use_GAN==1 then
           local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
           local label = torch.FloatTensor(output[{{},1}]:size()):fill(real_label) -- fake labels are real for generator cost
           if opt.gpu>0 then 
           	label = label:cuda();
           end
           local zero_batch=torch.Tensor(sum_score_D:size()):zero():cuda()
           local diff=ngen*score_D_cache[i]-sum_score_D-cosine_distance(feature_cache,i)
           diff=diff:cuda()
           diff= -diff/(ngen-1)
           local relu_diff=G.relu:forward(diff)
           relu_diff=relu_diff:cuda()
           --errG = criterion:forward(output, label) 
           output=netD:forward(fake_AB[i])
           errG=errG+criterion:forward(output,label) + compete_criterion:forward(relu_diff,zero_batch)   
           local compete_df_do=G.relu:backward( diff , compete_criterion:backward( relu_diff , zero_batch )  )
           compete_df_do=compete_df_do:repeatTensor(opt.batchSize*1*30*30):cuda()
           compete_df_do=compete_df_do:reshape( output:size() )
           local df_do = criterion:backward(output, label) + opt.lambda_compete*compete_df_do
           df_dg = netD:updateGradInput(fake_AB[i], df_do):narrow(2,fake_AB[i]:size(2)-output_nc+1, output_nc)
           
        else
            errG = 0
        end
        
        -- unary loss
        local df_do_AE = torch.zeros(fake_B[i]:size())
        if opt.gpu>0 then 
        	df_do_AE = df_do_AE:cuda();
        end
        if opt.use_L1==1 then
           errL1 = criterionAE:forward(fake_B[i], real_B)
           df_do_AE = criterionAE:backward(fake_B[i], real_B)
        else
            errL1 = 0
        end
        
        G['netG'..i]:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
    end 
    errG=errG/ngen
    return errG, gradParametersG
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
        
        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)
        
        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                --disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+1, title=opt.name .. ' target'})
                for i6 = 1,ngen do
                    local fake_B_s = util.scaleBatch(fake_B[i6]:float(),100,100)
                    disp.image(util.deprocess_batch(util.scaleBatch(fake_B_s:float(),100,100)), {win=opt.display_id+1+i6, title=opt.name .. ' output'..i6})
                end
           else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
                --disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' target'})
                for i6 = 1,ngen do
                    disp.image(util.deprocess_batch(util.scaleBatch(fake_B[i6]:float(),100,100)), {win=opt.display_id+1+i6, title=opt.name .. ' output'..i6})
                end
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                --for i5 = 1,ngen do
                    --dfake_B=fake_B[i5]
                    if opt.preprocess == 'colorization' then 
                        for i2=1, opt.batchSize do
                            fakes=nil
                            for i5=1,ngen do
                                if fakes==nil then
                                    fakes=util.deprocessLAB(real_A[i2]:float(), fake_B[i5][i2]:float())
                                else
                                    fakes=torch.cat(fakes,util.deprocessLAB(real_A[i2]:float(), fake_B[i5][i2]:float()),3)
                                end
                            end
                            if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),fakes,3)/255.0
                            else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),fakes,3)/255.0, 2) end
                        end
                    else
                        for i2=1, opt.batchSize do
                            fakes=nil
                            for i5=1,ngen do
                                if fakes==nil then
                                    fakes=util.deprocess(fake_B[i5][i2]:float())
                                else
                                    fakes=torch.cat(fakes,util.deprocess(fake_B[i5][i2]:float()),3)
                                end
                            end
                            if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),fakes,3)
                            else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),fakes,3), 2) end
                        end
                    end
                --end
                image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            end
            
            opt.serial_batches=serial_batches
        end
        
        -- logging
        if counter % opt.print_freq == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                     epoch, ((i-1) / opt.batchSize),
                     math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG and errG or -1, errD and errD or -1, errL1 and errL1 or -1))
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), G)
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD)
        end
        
    end
    
    
    --parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    --parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), G)
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD)
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    --parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    --parametersG, gradParametersG = netG:getParameters()
    --parametersG, gradParametersG =model_utils.combine_all_parameters(G)
end
