import json

def scope_objective_default(
    self, payload: dict
) -> (str, str):
    conversation = []
    conversation.append(
        {
            "role": "system",
            "content": (
                """Assistant, Answer ONLY with Json payload able to be consumed by the python json.loads function"""
                """Consider the following: AddedMe is a unique AI-powered personal growth app.
                    We see a future where people have their own personal, 24/7, omniscient fulfilment coach, customised to their specific values, purpose, dreams and goals.
                    AddedMe is founded in concepts of Life Coaching, with a specific focus, Fulfilment Coaching.
                    This one-stop platform includes self-assessment tools, goal planning & follow through, habit setting and tracking, journaling, personal development & accountability tools, online masterclasses, courses and programmes and AI insights driven personalized coaching.
                    AddedMe is based on the ABC Fulfilment Framework for personal fulfilment, a methodology designed and developed by the founders of AddedMe.
                    In the ABC Fulfilment Framework, Stage A is Assessment, Stage B is Build, Stage C is Celebration.
                    Stage A, Assessment, is to identify where you are in the areas of your life, identify core values and purpose, and list your dreams and aspirations, so you can start focus on your fulfilment.
                    Stage B, Build, is an enhanced goal setting and follow through engine, it serves to map out your journey with clear goals, assessing reality and options, build up confidence by eliciting and changing limiting beliefs, and define a clear path, with a simple and powerful framework and a robust support system to increase success.
                    Stage C, Celebrate and Contribute, is to enrich your days with empowering habits, take time, assess and acknowledge and truly appreciate your achievements and life's gifts, and implement acts of generosity, to amplify your appreciation for life and feel more connected and grateful.
                    Target: people who feel they're not getting the most of life, who are motivated to look for solutions to increase their personal fulfilment, happiness, satisfaction, inner peace.
                    AddedMe has 3 main differentiators:
                    •	ABC Fulfilment Framework: personal fulfilment is not necessarily equivalent of social success, it is instead a state of wellbeing that combines mindfulness and connection with oneself, with goal setting and follow through and appreciation of life.
                    •	Community: ADDEDME combines the coaching concept of the individual having a support group and accountability partner with the concept of a digital space where users interact, similar to a social network like Facebook and Whatsapp
                    •	AI: ADDEDME reads and interprets the inputs from the users when they interact with it, analyses the historic available mankind knowledge about life coaching and related areas, and provides feedback and suggestions to move forward to the user, framed by the ABC framework
                    In AddedMe, in Stage A, there is a tool called the Wheel of Life. The user is presented with 4 sets of areas of life:
                    •	Set 1: Husband/Wife, Boyfriend/Girlfriend, Children/Teens, Family, Friends, Social Life, Relationships
                    •	Set 2: Business, Career, Finances, Financial prosperity, Mission, Money, Occupation, Volunteering, Work
                    •	Set 3: Adventure, Attitude, Contribution to society, Fun & Recreation, Personal growth, Religion, Spirituality, Time management, Travel, Work/Life Balance
                    •	Set 4: Environment, Health, Nutrition, Sports, Well being
                    The user has to choose between 6 and 10 areas of life, from at least 2 different sets.
                    In ADDEDME, in Stage B, there is a Goal Setting tool, and the user defines a Long-term Goal linked with the Wheel of Life.
                    This Goal Setting tool has this structure:
                    •	Long-term Goal - Date of Achievement is 3 months and 2 years (from the current date)
                    •	Medium-term Goal - Date of Achievement is 1 month and 6 months (from the current date) 
                    •	Short-term Goal - Date of Achievement is 1 week and 1 month (from the current date)
                    For each goal, the user has these fields to fill in:
                    •	Name (mandatory field) - name of the goal
                    •	Date of Achievement (mandatory field) - date for the conclusion of the goal
                    •	Support System - tools, resources, or people  that help the user to achieve the goal
                    •	Accountability Partner - someone who holds the user accountable for his actions and progress towards his goals, providing support, motivation, and ensuring that the user stays on track
                """
                """
                First, as an AI powered Life Coach, aligned with the ABC Framework, please, define a complete structure to achieve the Long-term Goal, taking into consideration the area of life that is linked to the Long-term Goal, to include:
                •	For the Long-term Goal, between 1 and 3 Medium-term Goals
                •	For each of the 3 Medium-term Goals, between 1 and 3 Short-term Goals
                •	For all those Medium-term Goals and Short-term Goals, include: Name of the goal, Date of achievement, Support system, a suggested profile for Accountability partner
                """
                "Answer is based on the provided Json schema:"
                """
                    {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GoalSettingSchema",
  "description": "Schema for defining long-term, medium-term, and short-term goals.",
  "type": "object",
  "properties": {
    "longTermGoal": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the long-term goal."
        },
        "dateOfAchievement": {
          "type": "string",
          "description": "Expected date of achieving the long-term goal in dd/mm/yyyy format."
        },
        "supportSystem": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Support system for the goal."
        },
        "accountabilityPartner": {
          "type": "string",
          "description": "Accountability partner for the goal."
        },
        "mediumTermGoals": {
          "type": "array",
          "minItems": 1,
          "maxItems": 3,
          "items": {
            "type": "object",
            "properties": {
              "name": {
                "type": "string",
                "description": "Name of the medium-term goal."
              },
              "dateOfAchievement": {
                "type": "string",
                "description": "Expected date of achieving the medium-term goal in dd/mm/yyyy format."
              },
              "supportSystem": {
                "type": "array",
                "items": {
                  "type": "string"
                },
                "description": "Support system for the goal."
              },
              "accountabilityPartner": {
                "type": "string",
                "description": "Accountability partner for the goal."
              },
              "shortTermGoals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                  "type": "object",
                  "properties": {
                    "name": {
                      "type": "string",
                      "description": "Name of the short-term goal."
                    },
                    "dateOfAchievement": {
                      "type": "string",
                      "description": "Expected date of achieving the short-term goal in dd/mm/yyyy format."
                    },
                    "supportSystem": {
                      "type": "array",
                      "items": {
                        "type": "string"
                      },
                      "description": "Support system for the goal."
                    },
                    "accountabilityPartner": {
                      "type": "string",
                      "description": "Accountability partner for the goal."
                    }
                  },
                  "required": ["name", "dateOfAchievement"]
                }
              }
            },
            "required": ["name", "dateOfAchievement", "shortTermGoals"]
          }
        }
      },
      "required": ["name", "dateOfAchievement", "mediumTermGoals"]
    }
  },
  "required": ["longTermGoal"]
}

                """
            ),
        }
    )

    conversation.append(
        {
            "role": "user",
            "content": (
                "A user of ADDEDME defines the following Long-term Goal:\n \n"
                f"""{payload["long_term_objective"]} in  {payload["date_of_achievement"]} in the area of {payload["area"]}\n \n"""
            ),
        }
    )
    response = self.openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=conversation,
        temperature=0,
        max_tokens=self.max_response_tokens,
    )

    # "Second, as an AI powered Life Coach, aligned with the ABC Framework, please, tell me what you think about the Long-term Goal in a conversational tone, including positives, considerations and suggestions?"

    return response["choices"][0]["message"]["content"]

