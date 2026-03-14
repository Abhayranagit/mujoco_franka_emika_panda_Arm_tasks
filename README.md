# mujoco_franka_emika_panda_Arm_tasks

 these are the project files in which i collect the task's demonstration performed by the franka emika panda arm in the mujoco simualtion . for the IK i have  used a library named as [pyroki](https://github.com/chungmin99/pyroki) which i gave a reference and you can also check there repo in the github . the library helps me to perform the task through franka emika panda arm in the given trajectories using the best possible joint angles . 
 
 
 Python version : 3.10.19





# Youtube videos 


<!-- BEGIN YOUTUBE-CARDS -->
[![USING VLM TO  PERFORM THE SATCKING TASK IN THE MUJOCO SIMULATION .](https://ytcards.demolab.com/?id=cmCszyCYurM&title=USING+VLM+TO++PERFORM+THE+SATCKING+TASK+IN+THE+MUJOCO+SIMULATION+.&lang=en&timestamp=1772719802&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "USING VLM TO  PERFORM THE SATCKING TASK IN THE MUJOCO SIMULATION .")](https://www.youtube.com/watch?v=cmCszyCYurM)
[![MuJoCo || TRANSFER TASK || FRANKA EMIKA PANDA ARM || PYROKI || SIMUALTION](https://ytcards.demolab.com/?id=coAn7eIkGgY&title=MuJoCo+%7C%7C+TRANSFER+TASK+%7C%7C+FRANKA+EMIKA+PANDA+ARM+%7C%7C+PYROKI+%7C%7C+SIMUALTION&lang=en&timestamp=1771407023&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "MuJoCo || TRANSFER TASK || FRANKA EMIKA PANDA ARM || PYROKI || SIMUALTION")](https://www.youtube.com/watch?v=coAn7eIkGgY)
[![MUJOCO || STACKING TASK || FRANKA EMIKA PANDA ARM || PYROKI LIBRARY ||  ROBOTICS || SIMULATION](https://ytcards.demolab.com/?id=ILbPJ6nKtbU&title=MUJOCO+%7C%7C+STACKING+TASK+%7C%7C+FRANKA+EMIKA+PANDA+ARM+%7C%7C+PYROKI+LIBRARY+%7C%7C++ROBOTICS+%7C%7C+SIMULATION&lang=en&timestamp=1770966014&background_color=%230d1117&title_color=%23ffffff&stats_color=%23dedede&max_title_lines=1&width=250&border_radius=5 "MUJOCO || STACKING TASK || FRANKA EMIKA PANDA ARM || PYROKI LIBRARY ||  ROBOTICS || SIMULATION")](https://www.youtube.com/watch?v=ILbPJ6nKtbU)
<!-- END YOUTUBE-CARDS -->


## Screenshots



<!-- or with specific size -->
<img src="Screenshot%202026-02-23%20150803.png" alt="Franka Arm Simulation" width="600"/>
**Figure:** Extracted frame from recorded simulation data at a specific timestep. The visualization displays the Franka Emika Panda arm state along with joint positions, action values, and comparison between actual and target values for each joint during the task execution.


## VLM integration
In stacking_task_v7 module using gemini-2.5 flash for the visual language model . Image of the current state for the environment is being fed to the VLm model. IN context to the prompt(task description/constraint ) VLM makes command to perform the  task. 


## Copilot Setup – Enabling Claude Opus Models in VS Code

If Claude Opus (or other non-default models) are not showing in the VS Code Copilot model picker, follow these steps:

### Requirements
1. **GitHub Copilot Pro** subscription or higher (Copilot Business / Enterprise with model access enabled by an organization admin). The free Copilot tier does not include access to Claude models.
2. **VS Code 1.99+** (or later) – older versions may not support the model picker.
3. **GitHub Copilot extension** (`github.copilot`) and **GitHub Copilot Chat extension** (`github.copilot-chat`) installed and up to date.

### How to Select Claude Opus
1. Open the **Copilot Chat** panel in VS Code (click the Copilot icon in the sidebar or press `Ctrl+Shift+I` / `Cmd+Shift+I`).
2. At the top of the chat panel, click the **model picker dropdown** (it shows the current model name, e.g. "GPT-4o").
3. Select **Claude 3.5 Sonnet**, **Claude 4 Opus**, or another available Claude model from the list.

### Troubleshooting
| Problem | Solution |
|---------|----------|
| No model picker visible | Update VS Code and both Copilot extensions to the latest version. |
| Claude models not listed | Verify your GitHub Copilot subscription includes model selection (Pro or higher). Go to [github.com/settings/copilot](https://github.com/settings/copilot) to check. |
| Organization policy blocks models | Ask your organization admin to enable "Allow model selection" under the Copilot policy settings at the org level. |
| Models greyed out | Sign out and sign back in to GitHub in VS Code (`GitHub: Sign Out` then `GitHub: Sign In` from the Command Palette). |
| Still not working | Try running `Developer: Reload Window` from the Command Palette, or reinstall the Copilot extensions. |
