// AWS Amplify for hosting frontend web app to query credit scores

variable "github_repository" {
  description = "GitHub repository URL for frontend app (e.g., https://github.com/username/repo)"
  type        = string
  default     = ""
}

variable "github_access_token" {
  description = "GitHub personal access token for Amplify to pull code"
  type        = string
  default     = ""
  sensitive   = true
}

variable "amplify_branch" {
  description = "Git branch to deploy"
  type        = string
  default     = "main"
}

// IAM role for Amplify
resource "aws_iam_role" "amplify_role" {
  name = "${var.project_name}-amplify-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "amplify.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "amplify_backend_deployment" {
  role       = aws_iam_role.amplify_role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess-Amplify"
}

// Amplify App
resource "aws_amplify_app" "credit_score_frontend" {
  name       = "${var.project_name}-frontend-${var.environment}"
  # Repository will be connected manually via Amplify Console
  # repository = var.github_repository != "" ? var.github_repository : null
  # access_token = var.github_access_token != "" ? var.github_access_token : null

  iam_service_role_arn = aws_iam_role.amplify_role.arn

  # Build settings for React/Vue/Next.js app
  build_spec = <<-EOT
    version: 1
    frontend:
      phases:
        preBuild:
          commands:
            - npm ci
        build:
          commands:
            - npm run build
      artifacts:
        baseDirectory: build
        files:
          - '**/*'
      cache:
        paths:
          - node_modules/**/*
    backend:
      phases:
        build:
          commands:
            - echo "Backend build not needed - using API Gateway"
  EOT

  # Environment variables for frontend to access API
  environment_variables = {
    API_ENDPOINT           = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
    API_GATEWAY_URL        = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}/predict"
    REACT_APP_API_ENDPOINT = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
    VUE_APP_API_ENDPOINT   = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
    NEXT_PUBLIC_API_ENDPOINT = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
  }

  # Enable auto branch creation
  enable_auto_branch_creation = false
  enable_branch_auto_build    = true
  enable_branch_auto_deletion = false

  # Custom rules for SPA routing
  custom_rule {
    source = "/<*>"
    status = "404"
    target = "/index.html"
  }

  custom_rule {
    source = "</^[^.]+$|\\.(?!(css|gif|ico|jpg|js|png|txt|svg|woff|woff2|ttf|map|json)$)([^.]+$)/>"
    status = "200"
    target = "/index.html"
  }

  tags = {
    Name        = "${var.project_name}-frontend"
    Environment = var.environment
  }
}

// Amplify Branch
resource "aws_amplify_branch" "main" {
  app_id      = aws_amplify_app.credit_score_frontend.id
  branch_name = var.amplify_branch

  enable_auto_build = true
  stage             = var.environment == "prod" ? "PRODUCTION" : "DEVELOPMENT"

  environment_variables = {
    API_ENDPOINT = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
  }
}

// Amplify Domain (optional - for custom domain)
# resource "aws_amplify_domain_association" "custom_domain" {
#   app_id      = aws_amplify_app.credit_score_frontend.id
#   domain_name = "yourdomain.com"
#
#   sub_domain {
#     branch_name = aws_amplify_branch.main.branch_name
#     prefix      = var.environment == "prod" ? "" : var.environment
#   }
# }

// Outputs
output "amplify_app_url" {
  value       = "https://${var.amplify_branch}.${aws_amplify_app.credit_score_frontend.default_domain}"
  description = "URL of deployed Amplify app"
}

output "amplify_app_id" {
  value       = aws_amplify_app.credit_score_frontend.id
  description = "Amplify App ID"
}

output "amplify_console_url" {
  value       = "https://console.aws.amazon.com/amplify/home?region=${var.aws_region}#/${aws_amplify_app.credit_score_frontend.id}"
  description = "Amplify Console URL for managing deployments"
}

output "frontend_setup_instructions" {
  value = <<-EOT
    Frontend Setup Instructions:

    STEP 1: Create GitHub Repository
    ---------------------------------
    1. Create a GitHub repository for your frontend app
    2. Initialize a React/Vue/Next.js app locally:
       - React: npx create-react-app credit-score-frontend
       - Vue: npm create vue@latest credit-score-frontend
       - Next.js: npx create-next-app credit-score-frontend

    3. Add API integration code to call:
       https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}/predict

    4. Example React component (src/CreditScoreChecker.js):
       ```javascript
       import React, { useState } from 'react';

       function CreditScoreChecker() {
         const [customerId, setCustomerId] = useState('');
         const [result, setResult] = useState(null);
         const [loading, setLoading] = useState(false);
         const API_ENDPOINT = process.env.REACT_APP_API_ENDPOINT;

         const checkScore = async () => {
           setLoading(true);
           try {
             const response = await fetch('$${API_ENDPOINT}/predict', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ customer_id: customerId })
             });
             const data = await response.json();
             setResult(data);
           } catch (error) {
             console.error('Error:', error);
             setResult({ error: error.message });
           }
           setLoading(false);
         };

         return (
           <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
             <h1>VPBank Credit Score Checker</h1>
             <div style={{ marginBottom: '20px' }}>
               <input
                 type="text"
                 value={customerId}
                 onChange={(e) => setCustomerId(e.target.value)}
                 placeholder="Enter Customer ID"
                 style={{ padding: '10px', width: '70%%', fontSize: '16px' }}
               />
               <button
                 onClick={checkScore}
                 disabled={loading || !customerId}
                 style={{ padding: '10px 20px', marginLeft: '10px', fontSize: '16px' }}
               >
                 {loading ? 'Checking...' : 'Check Score'}
               </button>
             </div>
             {result && (
               <div style={{
                 padding: '20px',
                 backgroundColor: '#f5f5f5',
                 borderRadius: '8px',
                 marginTop: '20px'
               }}>
                 <h2>Results:</h2>
                 <pre>{JSON.stringify(result, null, 2)}</pre>
               </div>
             )}
           </div>
         );
       }

       export default CreditScoreChecker;
       ```

    5. Push code to GitHub:
       git init
       git add .
       git commit -m "Initial commit"
       git remote add origin https://github.com/yourusername/credit-score-frontend.git
       git push -u origin main

    STEP 2: Connect Amplify to GitHub
    ----------------------------------
    1. Go to Amplify Console:
       https://console.aws.amazon.com/amplify/home?region=${var.aws_region}#/${aws_amplify_app.credit_score_frontend.id}

    2. Click "Connect repository" or "Connect branch"

    3. Choose "GitHub" as the repository service

    4. Authorize AWS Amplify to access your GitHub account

    5. Select your repository and branch (main)

    6. Amplify will auto-detect the build settings. Verify:
       - Build command: npm run build
       - Base directory: (leave empty)
       - Build output directory: build

    7. The environment variables are ALREADY configured:
       ✓ REACT_APP_API_ENDPOINT = ${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}
       ✓ VUE_APP_API_ENDPOINT = (same)
       ✓ NEXT_PUBLIC_API_ENDPOINT = (same)

    8. Click "Save and deploy"

    STEP 3: Access Your App
    ------------------------
    After deployment completes, your app will be available at:
    https://main.${aws_amplify_app.credit_score_frontend.default_domain}

    Auto-deployment: Every git push to main will trigger automatic deployment!

    Alternative: Manual Deployment via Amplify Console
    ---------------------------------------------------
    If you don't want to connect GitHub, you can:
    1. Build locally: npm run build
    2. Drag & drop the build folder to Amplify Console
    3. Or use Amplify CLI: amplify publish
  EOT
  description = "Instructions for setting up frontend app with Amplify"
}
