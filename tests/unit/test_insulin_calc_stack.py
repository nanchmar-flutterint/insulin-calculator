import aws_cdk as core
import aws_cdk.assertions as assertions

from insulin_calc.insulin_calc_stack import InsulinCalcStack


# example tests. To run these tests, uncomment this file along with the example
# resource in insulin_calc/insulin_calc_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = InsulinCalcStack(app, "insulin-calc")
    template = assertions.Template.from_stack(stack)


#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
