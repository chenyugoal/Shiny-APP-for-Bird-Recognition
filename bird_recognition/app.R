library(shiny)
library(wordcloud)
library(DT)
library(keras)

# Functions
# Get the index of class with highest possibilities
index.top.N = function(xs, N=10){
    if(length(xs) > 0) {
        o = order(xs, na.last=FALSE)
        o.length = length(o)
        if (N > o.length) N = o.length
        rev(o[((o.length-N+1):o.length)])
    }
    else {
        0
    }
}

# Load model
model <- load_model_hdf5("./model/birds_classification.h5")
TOP_CLASSES            <- 5
IMAGE_FORMAT <- c(224, 224)

# Load Species Name
loc_class_file <- "./classes.txt" 
classes <- read.csv(loc_class_file, header=FALSE, sep="")
classes <- classes[[2]] %>% strsplit(split = '[.]')
class = c()
for (i in classes){
    class = c(class,i[[2]])
}

# Header
get_header <- function() {
    HTML("<!-- common header-->
        <div id='headerSection'>
          <h1>Bird Recognition App</h1>
          <span style='font-size: 1.2em'>
            <span>Created by </span>
            <a>Chenyu Gao (cgao17@jhu.edu)</a>
            &bull;
            <span>May 2021</span>
          </span>
        </div>")
}


# Define UI for application that draws a histogram
ui <- fluidPage(
    tags$head(tags$link(rel = "stylesheet", type = "text/css", href = "style.css")),
    get_header(),          
    tabsetPanel(
        tabPanel("Image upload / Wordcloud",
                 fluidRow(
                     column(width=4, fileInput('file', '',accept = c('.jpg','.jpeg'))),
                     column(width=4, tags$label(HTML("&zwnj;")), tags$br(), tags$em("Please use the browse button to upload an image (JPG/JPEG format)"))
                 ),
                 tags$hr(),
                 fluidRow(
                     column(width=4, imageOutput('outputImage')),
                     column(width=4, plotOutput("plot"))
                 )),
        tabPanel("Predicted classes & scores", 
                 tags$p(),
                 fluidRow(column(width=8, dataTableOutput("table")))),
        tabPanel("About",
                 img(src = "collage.jpg", height = 320, width = 640),
                 p("This is a shiny app for bird recognition using convolutional neural network (CNN). 
                         Please do not attempt to sell the app for commercial use because the accuracy is still unsatisfying. 
                         I will optimize the model and re-publish it in the future."),
                 h3("Dataset & Model"),
                 p("Caltech-UCSD Birds-200-2011 (CUB-200-2011) is used as the dataset. 
                         It contains 11,788 images of 200 bird species, including Johns Hopkins's mascot, Blue Jay!!!!!!"),
                 p("To form the classification model, the Resnet-50 and two fully conneted layers are concatenated together. 
                         The Resnet-50 was pretrained on ImageNet by Keras. All of its parameters are freezed during training. 
                         Data augmentation techniques are used, including rotation, shifting, shearing, zooming, flipping, to avoid overfitting."),
                 p("On the validation set (20% of the dataset), the model achieves an accuracy of ~ 0.35, which is far lower than those of the benchmarks. 
                 The simple and shallow structure of the fully connected layers is to blame.
                 I will take some deep learning courses to build a more robust model in the future."),
                 h3("References"),
                 p("1. https://gerinberg.com/2019/12/10/image-recognition-keras/"),
                 p("2. https://www.r-bloggers.com/2018/02/deep-learning-image-classification-with-keras-and-shiny/"),
                 p("3. https://keras.io/api/applications/")
        )
    )

)

# Define server logic required to draw a histogram
server <- function(input, output) {
    
    outputtext <- reactive({
        req(input$file)
        
        img <- image_load(input$file$datapath, target_size = IMAGE_FORMAT)
        x <- image_to_array(img)
        x <- array_reshape(x, c(1, dim(x)))
        preds <- model %>% predict(x)
        # Index of the top classes
        ind <- index.top.N(preds, N=TOP_CLASSES)
        df <- data.frame(
            Object = class[ind], 
            Score = preds[ind]
        )
        df
    })
    
    output$plot <- renderPlot({
        df <- outputtext()
        # Separate long categories into shorter terms, so that we can avoid "could not be fit on page. It will not be plotted" warning as much as possible
        objects <- strsplit(as.character(df$Object), ',')
        df <- data.frame(Object = unlist(objects), 
                         Score  = rep(df$Score, sapply(objects, FUN = length)))
        wordcloud(df$Object, df$Score, scale = c(4,2),
                  colors = brewer.pal(6, "RdBu"), random.order = F)
    })
    
    output$outputImage <- renderImage({
        req(input$file)
        
        outfile <- input$file$datapath
        contentType <- input$file$type
        list(src = outfile,
             contentType=contentType,
             width = 400)
    }, deleteFile = TRUE)
    
    output$table <- renderDataTable({
        DT::datatable(outputtext(), 
                      rownames = FALSE,
                      options = list(pageLength = TOP_CLASSES, dom = 't'))
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
