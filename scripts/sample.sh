
CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
 --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
 --vid-path assets/airplane_vgg.mp4 \
 --text-prompt "An airplane flies overhead" 

# CUDA_VISIBLE_DEVICES=3 python sample_VT_comb.py ODE \
#  --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
#  --vid-path assets/airplane_vgg.mp4 \
#  --text-prompt "An airplane flies overhead and an ambulance siren blares" \
#  --save-suffix _ambulance

# CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
#  --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
#  --vid-path assets/otter_sora.mp4 \
#  --text-prompt "An otter growls and soft wind is blowing."

# CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
#  --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
#  --vid-path assets/suv_sora.mp4 \
#  --text-prompt "A car races down the road and police siren blares."

# CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
#  --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
#  --vid-path assets/flute_vgg.mp4 \
#  --text-prompt "A person plays a flute and an acoustic guitar is strummed."

# CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
#  --cross-att-vis-text --s-vis 7.5 --s-txt 2.5 \
#  --vid-path assets/dragon_pixverse.mp4 \
#  --text-prompt "A loud sound of fire explosion." \
#  --save-suffix _fire