"""
Gradio template borrowed from FoleyCrafter: https://github.com/open-mmlab/foleycrafter
"""

import os
import random
import gradio as gr
import torch
import torch
import warnings
import soundfile as sf

from demo.utils import get_parser, get_vid_name, get_save_text, \
    replace_audio, vid_feat_dir_exists, download_model
from train_utils import parse_ode_args, parse_transport_args
from models_VT_comb import SiT_models
from download import find_model
from aud_utils import get_vae_stft
from transport import create_transport, Sampler
from visual_feats.compute_vid_feats import compute_visual_features

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

parser = get_parser()
parse_transport_args(parser)
parse_ode_args(parser)

args = parser.parse_known_args()[0]
args.aud_size = [1024,64]
args.latent_size = [256,16]
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.cross_att_vis_text=True
args.demo=True
args.use_cache=True

class AudioGenerator:
    def __init__(self):
        # config dirs
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'demo_audio'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'demo_video'), exist_ok=True)
        vid_feat_dir_exists(args.vid_feat_dir)
        self.load_model()

    def load_model(self):
        gr.Info("Start Load Models...")
        download_model()
        self.model = SiT_models[args.model](
            input_size=args.latent_size,
            use_masking=args.use_masking,
            vid_feat_dir = args.vid_feat_dir,
            cross_att_vis_text=args.cross_att_vis_text,
            demo=args.demo,
        ).to(args.device)
        state_dict = find_model(args.ckpt)
        if args.txt_wt > 0:
            for key in state_dict.keys():
                if "weight_text" in key:
                    state_dict[key] = torch.full_like(state_dict[key], args.txt_wt) 
        self.model.load_state_dict(state_dict)
        self.model.eval()  # important!

        self.vae, _ = get_vae_stft()

        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps
        )
        self.sampler = Sampler(transport)
        
    def generate(
        self,
        vid_path,
        prompt_text,
        sample_steps,
        seed,
    ):
        args.num_sampling_steps = sample_steps
        args.seed = seed
        torch.manual_seed(args.seed)
        wts_list = [(7.5, 7.5), (2.5, 7.5), (7.5, 2.5), (2.5, 2.5)]
        TEXT_NULL = ""; VID_NULL = "EMPTY"
        vid = get_vid_name(vid_path)
        n = len(wts_list)

        self.sample_fn = self.sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse
        )

        save_txt = get_save_text(prompt_text)
        if args.use_cache:
            curr_vid_path = os.path.join(args.save_dir, 'demo_video', f"{vid}_{save_txt}_Svis{wts_list[0][0]}_Stxt{wts_list[0][1]}_{seed}.mp4")
            if os.path.exists(curr_vid_path):
                gr.Info(f"Skipping {curr_vid_path} as it already exists.")
                output_paths = []
                for j in range(len(wts_list)):
                    save_vid_path = os.path.join(args.save_dir, 'demo_video', f"{vid}_{save_txt}_Svis{wts_list[j][0]}_Stxt{wts_list[j][1]}_{seed}.mp4")
                    output_paths.append(save_vid_path)
                return output_paths

        compute_visual_features(vid_path, vid, args.vid_feat_dir, recalculate=args.recalculate_feat)
        
        
        zs = torch.randn(n, 8, args.latent_size[0], args.latent_size[1], device=args.device)
        zs = torch.cat([zs,zs,zs], 0)

        v_list=[]; t_list=[]; s_vis=[]; s_txt=[]
        for i in range(n):
            v_list.append(vid)
            t_list.append(prompt_text)
            s_vis.append(wts_list[i][0])
            s_txt.append(wts_list[i][1])

        t_list = [TEXT_NULL] * n + t_list + [TEXT_NULL] * n
        v_list = v_list + v_list + [VID_NULL] * n           

        model_kwargs = dict(y_txt_in=t_list, y_vid_in=v_list, s_vis=s_vis, s_txt=s_txt)
        # model_kwargs = dict(y_txt_in=t_list, y_vid_in=v_list, s_vis=2.5, s_txt=2.5)
        with torch.no_grad():
            samples = self.sample_fn(zs, self.model.forward_with_cfg_VT_demo, **model_kwargs)[-1]
            # samples = self.sample_fn(zs, self.model.forward_with_cfg_VT, **model_kwargs)[-1]
            samples, _, _ = samples.chunk(3, dim=0)  # Remove null class samples
        
        mel = self.vae.decode_first_stage(samples)
        wave = self.vae.decode_to_waveform(mel)

        all_outputs = [item for item in wave]
        output_paths = []
        for j, wav in enumerate(all_outputs):
            sav_sr = 16000
            save_aud_path = os.path.join(args.save_dir, 'demo_audio', f"{vid}_{save_txt}_Svis{wts_list[j][0]}_Stxt{wts_list[j][1]}_{seed}.wav")
            sf.write(save_aud_path, wav, samplerate=sav_sr)

            save_vid_path = os.path.join(args.save_dir, 'demo_video', f"{vid}_{save_txt}_Svis{wts_list[j][0]}_Stxt{wts_list[j][1]}_{seed}.mp4")
            replace_audio(vid_path, save_aud_path, save_vid_path)
            output_paths.append(save_vid_path)
        return output_paths[0], output_paths[1], output_paths[2], output_paths[3]
        # return output_paths[0]

controller = AudioGenerator()

with gr.Blocks(css=css) as demo:
    gr.HTML(
        '''
        <div style="text-align: center;">
            <div style="height: 66px; display: flex; align-items: center; justify-content: center; margin-bottom: 0;">
                <span style="height: 100%; width:66px;"></span>
                <strong style="font-size: 36px;">VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation</strong>
            </div>
            <div style="font-size: 24px; margin-top: 0;">
                <a href="https://sakshamsingh1.github.io/">Saksham Singh Kushwaha</a>,&nbsp;
                <a href="https://www.yapengtian.com/">Yapeng Tian</a><br>
                <span>The University of Texas at Dallas</span>
            </div>
        </div>
        '''
    )
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='5'><a href='https://sakshamsingh1.github.io/vintage/'>Project Page</a> &ensp;"  # noqa
            "<a href='https://arxiv.org/pdf/2412.10768'>Paper</a> &ensp;"
            "<a href='https://github.com/sakshamsingh1/vintage_aud_gen'>Code</a> &ensp;"
        )

    with gr.Column(variant="panel"):
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    input_video = gr.Video(label="Input Video")
                with gr.Row():
                    prompt_textbox = gr.Textbox(value="", label="Prompt", lines=1)

                with gr.Accordion("Sampling Settings", open=False):
                    with gr.Row():
                        sample_step_slider = gr.Slider(label="Sampling steps", value=250, minimum=100, maximum=500, step=50)

                with gr.Row():
                    seed_textbox = gr.Textbox(label="Seed", value=42)
                    seed_button = gr.Button(value="\U0001f3b2", elem_classes="toolbutton")
                seed_button.click(fn=lambda x: random.randint(1, 1e8), outputs=[seed_textbox], queue=False)

                generate_button = gr.Button(value="Generate", variant="primary")

                with gr.Row():
                    gr.Markdown(
                        "<div style='word-spacing: 6px;'><font size='4'><b>Steps</b>: <br> \
                        1. Upload the video and enter prompt.<br> \
                        2. Press Generate.<br><br> \
                        Gradio template inspired from <a href='https://github.com/open-mmlab/FoleyCrafter' target='_blank'>FoleyCrafter</a></font></div>"
                    )

            with gr.Column():
                result_video_0 = gr.Video(label="Generated Audio: (S_vis=7.5, S_txt=7.5)", interactive=False)
                result_video_1 = gr.Video(label="Generated Audio (S_vis=2.5, S_txt=7.5)", interactive=False)
                result_video_2 = gr.Video(label="Generated Audio (S_vis=7.5, S_txt=2.5)", interactive=False)
                result_video_3 = gr.Video(label="Generated Audio (S_vis=2.5, S_txt=2.5)", interactive=False)

        generate_button.click(
            fn=controller.generate,
            inputs=[
                input_video,
                prompt_textbox,
                sample_step_slider,
                seed_textbox,
            ],
            outputs=[result_video_0, result_video_1, result_video_2, result_video_3],
            # outputs=[result_video_0],
        )

        gr.Examples(
            examples=[
                ["assets/airplane_vgg.mp4", "An airplane flies overhead and an ambulance siren blares", 250, "12345678"],
                ["assets/otter_sora.mp4", "An otter growls and soft wind is blowing.", 250, "87654321"],
                ["assets/suv_sora.mp4", "A car races down the road and police siren blares.", 250, "11223344"],
                ["assets/dragon_pixverse.mp4", "A dragon spits fire and sea waves are crashing. ", 250, "55667788"],
                ["assets/flute_vgg.mp4", "A person plays a flute and an acoustic guitar is strummed.", 250, "5567788"],
            ],
            inputs=[
                input_video,
                prompt_textbox,
                sample_step_slider,
                seed_textbox
            ],
            outputs=[result_video_0, result_video_1, result_video_2, result_video_3],
            cache_examples=True,
            fn=controller.generate,
        )

    # demo.queue(10)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
